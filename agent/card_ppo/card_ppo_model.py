"""Small candidate-shared actor/critic over public-card-selection-1 only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from sts2_env.gym_env.card_selection_observation import (
    CARD_NAMES, RELIC_NAMES, TYPE_NAMES, RARITY_NAMES, MAX_OFFERS, SCHEMA_HASH,
)

MODEL_VERSION = "card-candidate-actor-critic-1"
SHAPES = dict(run=(10,), deck=(len(CARD_NAMES),3), relics=(len(RELIC_NAMES),3), candidates=(16,10))


@dataclass(frozen=True)
class ModelConfig:
    card_embedding: int = 16
    relic_embedding: int = 8
    context: int = 64
    candidate_hidden: int = 64
    skip_hidden: int = 32

    def validate(self):
        if any(type(x) is not int or not 1 <= x <= 256 for x in asdict(self).values()):
            raise ValueError("Small positive architecture dimensions required")


def policy_digest(model: nn.Module) -> str:
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        t=t.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def observation_tensors(observation: Mapping, *, batched: bool = False) -> dict[str,torch.Tensor]:
    if set(observation)!=set(SHAPES):
        raise ValueError("Only the four declared public observation blocks are accepted")
    result={}
    for name,shape in SHAPES.items():
        t=torch.as_tensor(observation[name],dtype=torch.float32,device='cpu')
        expected=tuple(t.shape[1:]) if batched else tuple(t.shape)
        if expected!=shape or not torch.isfinite(t).all():
            raise ValueError(f"Invalid public feature block: {name}")
        result[name]=t if batched else t.unsqueeze(0)
    if len({t.shape[0] for t in result.values()})!=1:
        raise ValueError("Inconsistent observation batch dimensions")
    return result


def stack_observations(observations: list[dict]) -> dict[str,torch.Tensor]:
    return observation_tensors({k:np.stack([o[k] for o in observations]) for k in SHAPES},batched=True)


class CardActorCritic(nn.Module):
    """No slot-specific weights; exact slot ordering is retained at the output.

    Public deterministic scaling/pooling is learner preprocessing, not a schema
    change. Last actor layers start at zero: every legal action, including Skip,
    has the same initial probability regardless of identity/type/rarity.
    """
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__();config.validate();self.config=config
        self.card_embedding=nn.Embedding(len(CARD_NAMES)+1,config.card_embedding,padding_idx=0)
        self.relic_embedding=nn.Embedding(len(RELIC_NAMES),config.relic_embedding)
        self.type_embedding=nn.Embedding(len(TYPE_NAMES)+1,4,padding_idx=0)
        self.rarity_embedding=nn.Embedding(len(RARITY_NAMES)+1,3,padding_idx=0)
        context_in=10+3*config.card_embedding+3*config.relic_embedding
        self.context=nn.Sequential(nn.Linear(context_in,config.context),nn.Tanh(),nn.Linear(config.context,config.context),nn.Tanh())
        offer_size=config.card_embedding+4+3+6
        features=config.context+offer_size
        self.actor=nn.Sequential(nn.Linear(features,config.candidate_hidden),nn.Tanh(),nn.Linear(config.candidate_hidden,1))
        self.skip=nn.Sequential(nn.Linear(features,config.skip_hidden),nn.Tanh(),nn.Linear(config.skip_hidden,1))
        self.critic=nn.Sequential(nn.Linear(features,config.context),nn.Tanh(),nn.Linear(config.context,1))
        for head in (self.actor,self.skip,self.critic):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        self.register_buffer('run_scale',torch.tensor([100,100,1000,3,64,32,20,64,32,8],dtype=torch.float32))
        self.register_buffer('offer_scale',torch.tensor([1,5,5,50,50,10],dtype=torch.float32))

    @staticmethod
    def _tokens(t: torch.Tensor, maximum: int) -> torch.Tensor:
        if ((t<0)|(t>maximum)|(t!=t.round())).any():
            raise ValueError("Invalid categorical public token")
        return t.long()

    def forward(self, observation: Mapping, masks: torch.Tensor) -> tuple[Categorical,torch.Tensor]:
        obs=observation_tensors(observation,batched=True)
        masks=torch.as_tensor(masks,dtype=torch.bool,device='cpu')
        n=obs['run'].shape[0]
        if masks.shape!=(n,MAX_OFFERS+1) or not masks.any(dim=1).all():
            raise ValueError("Every sampled state needs a nonempty exact 17-action mask")
        c=obs['candidates'];present=c[:,:,0]>0.5
        if (masks[:,:16]&~present).any():
            raise ValueError("Legal candidate action points at padding")
        deck=torch.einsum('bci,ce->bie',obs['deck']/32,self.card_embedding.weight[1:]).flatten(1)
        relic=obs['relics']
        relic_features=torch.stack((relic[:,:,0],relic[:,:,1],relic[:,:,1]*relic[:,:,2]/10),dim=-1)
        relic_context=torch.einsum('bri,re->bie',relic_features/16,self.relic_embedding.weight).flatten(1)
        context=self.context(torch.cat((obs['run']/self.run_scale,deck,relic_context),dim=-1))
        offer=torch.cat((self.card_embedding(self._tokens(c[:,:,1],len(CARD_NAMES))),
                         self.type_embedding(self._tokens(c[:,:,7],len(TYPE_NAMES))),
                         self.rarity_embedding(self._tokens(c[:,:,8],len(RARITY_NAMES))),
                         c[:,:,[2,3,4,5,6,9]]/self.offer_scale),dim=-1)
        pooled=(offer*present.unsqueeze(-1)).sum(1)/present.sum(1,keepdim=True).clamp_min(1)
        global_features=torch.cat((context,pooled),dim=-1)
        candidate_features=torch.cat((context[:,None,:].expand(-1,16,-1),offer),dim=-1)
        logits=torch.cat((self.actor(candidate_features).squeeze(-1),self.skip(global_features)),dim=-1)
        values=self.critic(global_features).squeeze(-1)
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite actor/critic output")
        return Categorical(logits=logits.masked_fill(~masks,-torch.inf)),values

    @torch.no_grad()
    def sample(self, observation: dict, mask, generator: torch.Generator) -> tuple[int,float,float]:
        distribution,values=self(observation_tensors(observation),torch.as_tensor(mask,dtype=torch.bool)[None,:])
        action=torch.multinomial(distribution.probs,1,generator=generator).squeeze(-1)
        return int(action.item()),float(distribution.log_prob(action).item()),float(values.item())

    def architecture(self) -> dict:
        return dict(version=MODEL_VERSION,config=asdict(self.config),schema_sha256=SCHEMA_HASH,
            input_features=sum(int(np.prod(s)) for s in SHAPES.values()),actions=17,
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            trainable_names=[n for n,p in self.named_parameters() if p.requires_grad],preprocessing='fixed-public-scaling-pooling-1')
