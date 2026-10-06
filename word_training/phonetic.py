"""Whole-word BOT probability over collapsed CTC paths; no transcript output."""
import re
import numpy as np

# The only named letters are those of the target. All other speech is filler.
BLANK,B,O,T,SPACE,OTHER=range(6)
ALPHABET=['blank','б','о','т','boundary','other']


def targets(text):
    tokens=[]
    for character in ' '.join(re.findall(r'\w+',text.casefold())):
        token={'б':B,'о':O,'т':T,' ':SPACE}.get(character,OTHER)
        if token==OTHER and tokens and tokens[-1]==OTHER: continue
        tokens.append(token)
    return tokens


def keyword_probability(probabilities):
    """Sum paths containing a complete word, with CTC blank/repeat semantics.

    Parser states: start-of-word, B, BO, BOT, other-word, already-found.
    OTHER blocks suffix matches such as robot. BOT followed by OTHER rejects
    inflections such as botu. End of a complete utterance also ends a word.
    """
    value=np.asarray(probabilities,dtype=np.float64)
    single=value.ndim==2
    if single: value=value[None]
    if value.ndim!=3 or value.shape[-1]!=6: raise ValueError('Expected [batch,time,6] probabilities')
    if not np.isfinite(value).all() or (value<0).any(): raise ValueError('Invalid probabilities')
    total=value.sum(-1,keepdims=True)
    if (total<=0).any(): raise ValueError('Empty probability distribution')
    value=value/total
    # transitions[label, old parser state] -> new parser state
    transitions={B:[1,4,4,4,4,5],O:[4,2,4,4,4,5],T:[4,4,3,4,4,5],
                 SPACE:[0,0,0,5,0,5],OTHER:[4,4,4,4,4,5]}
    state=np.zeros((len(value),6,6),np.float64);state[:,0,BLANK]=1
    for frame in value.transpose(1,0,2):
        summed=state.sum(-1)
        following=np.zeros_like(state)
        following[:,:,BLANK]=summed*frame[:,BLANK,None]
        for token,mapping in transitions.items():
            # Repeated nonblank frames do not emit a second CTC character.
            following[:,:,token]+=state[:,:,token]*frame[:,token,None]
            emitted=(summed-state[:,:,token])*frame[:,token,None]
            for previous,destination in enumerate(mapping):
                following[:,destination,token]+=emitted[:,previous]
        state=following
    result=np.clip(state[:,3].sum(-1)+state[:,5].sum(-1),0,1)
    return float(result[0]) if single else result


def posterior(logits,temperature=1.):
    if temperature<=0: raise ValueError('Temperature must be positive')
    value=np.asarray(logits,dtype=np.float64)/temperature
    value=np.exp(value-value.max(-1,keepdims=True))
    return value/value.sum(-1,keepdims=True)


def forced_alignment(probabilities,transcript):
    """Viterbi-align known synthetic text; teacher errors cannot change labels."""
    probability=np.asarray(probabilities,dtype=np.float64)
    if probability.ndim!=2 or probability.shape[1]!=6:raise ValueError('Expected time by six states')
    if not transcript:return np.zeros(len(probability),np.int64)
    expanded=np.zeros(2*len(transcript)+1,np.int64);expanded[1::2]=transcript
    skip=np.zeros(len(expanded),bool)
    skip[2:]=(expanded[2:]!=BLANK)&(expanded[2:]!=expanded[:-2])
    score=np.full(len(expanded),-np.inf);score[0]=0
    trace=np.zeros((len(probability),len(expanded)),np.int8)
    log=np.log(np.maximum(probability,1e-12))
    for time,frame in enumerate(log):
        one=np.r_[-np.inf,score[:-1]]
        two=np.r_[-np.inf,-np.inf,score[:-2]]
        two[~skip]=-np.inf
        choices=np.stack([score,one,two])
        trace[time]=choices.argmax(0)
        score=choices.max(0)+frame[expanded]
    position=len(expanded)-1 if score[-1]>=score[-2] else len(expanded)-2
    if not np.isfinite(score[position]):raise ValueError('Transcript cannot fit the acoustic frames')
    path=np.zeros(len(probability),np.int64)
    for time in range(len(probability)-1,-1,-1):
        path[time]=expanded[position];position-=int(trace[time,position])
    return path
