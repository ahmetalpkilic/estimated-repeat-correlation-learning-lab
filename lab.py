"""Estimated AR correlation, acquisition allocation, and optical timing risk."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize_scalar
import optics as o
import clock_model as c
import repeat_model as r

LEVELS = (0., .4, .7, .9)
SEED = 20261012
ALLOCATIONS = {40: 10, 80: 8}  # 2*n characterization + 40*m calibration = 480


def reml_score(z, rho):
    """Profile matrix-normal REML: unknown 2-vector mean and 2x2 covariance.

    AR precision is tridiagonal, so applying it costs O(n), not a dense inverse.
    Ordinary centering improves numerical stability; Q removes any constant.
    Additive constants independent of rho are omitted from minus twice log L.
    """
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] != 2 or len(z) < 6 or not np.all(np.isfinite(z)):
        raise ValueError('Require finite (n>=6,2) characterization sequence')
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError('Require |rho|<1')
    n = len(z); z = z-z.mean(axis=0)
    diagonal = np.full(n, 1+rho*rho); diagonal[[0,-1]] = 1
    wz = diagonal[:,None]*z
    wz[1:] -= rho*z[:-1]; wz[:-1] -= rho*z[1:]
    wz /= 1-rho*rho
    w1 = diagonal.copy(); w1[1:] -= rho; w1[:-1] -= rho
    w1 /= 1-rho*rho
    v = z.T@w1
    scatter = z.T@wz - np.outer(v,v)/w1.sum()
    sign, logdet = np.linalg.slogdet(scatter/(n-1))
    if sign <= 0: raise ValueError('Characterization scatter must be full rank')
    return 2*(n-1)*np.log1p(-rho*rho)+2*np.log(w1.sum())+(n-1)*logdet


def fit_rho(z):
    """Bounded profile fit; retain endpoint solutions and report their frequency.

    The fixed [-.95,.95] interval is part of the estimator, not a CI. Both
    classes are estimated; knowing the even class is iid would give free info.
    """
    grid = np.linspace(-.95,.95,39)
    scores = np.array([reml_score(z,x) for x in grid])
    candidates = [(scores[0],grid[0]),(scores[-1],grid[-1])]
    for i in range(1,len(grid)-1):
        if scores[i] <= min(scores[i-1],scores[i+1]):
            fit = minimize_scalar(lambda x: reml_score(z,x), bounds=(grid[i-1],grid[i+1]),
                                  method='bounded', options={'xatol':1e-9})
            if not fit.success: raise RuntimeError('Correlation fit failed')
            candidates.append((fit.fun,fit.x))
    return float(min(candidates)[1])


def ar_sequence(normal, rho, cov, mean):
    """Independent stationary AR recursion, with marginal covariance cov."""
    e = np.empty_like(normal); e[0] = normal[0]
    for j in range(1,len(e)):
        e[j] = rho*e[j-1]+np.sqrt(1-rho*rho)*normal[j]
    return mean + e@np.linalg.cholesky(cov).T


def experiment(trials=300, seed=SEED):
    if not isinstance(trials,int) or trials < 2: raise ValueError('trials>=2 required')
    image_rng, eval_rng, cal_rng, char_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    cov = c.covariances(4.)
    means = np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))
    center = o.ORIGIN+o.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
    centers = [o.fit_center(o.image(center)+image_rng.normal(0,o.READ_SD,33)) for _ in range(trials)]
    evaluation = eval_rng.normal(size=(trials,40,2))
    calibration = cal_rng.normal(size=(trials,40,12,2))
    characterization = char_rng.normal(size=(trials,2,80,2))
    rows=[]
    for rho in LEVELS:
        true_rhos=np.array([0.,rho]); event_rhos=true_rhos[r.LABELS]
        errors={k:[] for k in ('oracle','naive12','known12','estimated40','known40','estimated80','known80')}
        estimates={n:[] for n in ALLOCATIONS}
        for k,u in enumerate(centers):
            z=means+np.einsum('nij,nj->ni',np.linalg.cholesky(cov),evaluation[k])
            repeats=r.draw_repeats(calibration[k],cov,event_rhos,means)
            # Characterization sequences are independent of repeat blocks and
            # evaluation. Their nuisance means/covariances are never supplied.
            char=[ar_sequence(characterization[k,g],true_rhos[g],cov[g],means[g]) for g in range(2)]
            policies={'oracle':cov, 'naive12':r.estimate(repeats,event_rhos,'naive'),
                      'known12':r.estimate(repeats,event_rhos)}
            for n,m in ALLOCATIONS.items():
                fitted=np.array([fit_rho(char[g][:n]) for g in range(2)])
                estimates[n].append(fitted.tolist())
                policies[f'estimated{n}']=r.estimate(repeats[:,:m],fitted[r.LABELS])
                policies[f'known{n}']=r.estimate(repeats[:,:m],event_rhos)
            for name,s in policies.items():
                fit=c.calibrate(*z.T,s)
                errors[name].append(1000*(c.delay_from_center(u,fit)-c.DELAY))
        paired={}
        for a,b in [('estimated40','naive12'),('estimated80','naive12'),('estimated40','known40'),('estimated80','known80'),('estimated80','estimated40')]:
            d=np.square(errors[a])-np.square(errors[b])
            paired[a+' minus '+b]=dict(difference_ms2=float(d.mean()),mcse_ms2=float(d.std(ddof=1)/np.sqrt(trials)))
        rows.append(dict(rho=rho, methods={name:dict(rmse_ms=float(np.sqrt(np.mean(np.square(e)))),bias_ms=float(np.mean(e)),sd_ms=float(np.std(e,ddof=1))) for name,e in errors.items()},
                         rho_estimates={str(n):dict(mean=np.mean(x,axis=0).tolist(),sd=np.std(x,axis=0,ddof=1).tolist(),boundary_fraction=np.mean(np.abs(x)>=.95-1e-7,axis=0).tolist()) for n,x in estimates.items()},paired_risk=paired))
    return dict(seed=seed,trials=trials,total_pairs=480,fresh_evaluation_pairs=40,
                allocations={str(n):dict(characterization_pairs=2*n,calibration_pairs=40*m,repeats=m) for n,m in ALLOCATIONS.items()},
                sigma_px=o.SIGMA,center_sd_px=float(np.std(centers,ddof=1)),failed_fits=0,rows=rows)


def demo(out='outputs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=experiment(); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    axes[0].plot(LEVELS,LEVELS,'k:',label='True rho')
    for n in ALLOCATIONS:
        axes[0].errorbar(LEVELS,[row['rho_estimates'][str(n)]['mean'][1] for row in result['rows']],yerr=[row['rho_estimates'][str(n)]['sd'][1] for row in result['rows']],fmt='o-',capsize=3,label=f'{n} pairs/class, ±1 SD')
    axes[0].set(title='Estimated correlation: noisier class',ylabel='Estimated rho')
    for name in ('oracle','naive12','known12','estimated40','estimated80'):
        axes[1].plot(LEVELS,[row['methods'][name]['rmse_ms'] for row in result['rows']],'o-',label=name)
    axes[1].set(title='Fresh optical-delay error',ylabel='Delay RMSE (ms)')
    for name in ('estimated40 minus naive12','estimated80 minus naive12','estimated80 minus estimated40'):
        axes[2].errorbar(LEVELS,[row['paired_risk'][name]['difference_ms2'] for row in result['rows']],yerr=[2*row['paired_risk'][name]['mcse_ms2'] for row in result['rows']],fmt='o-',capsize=3,label=name)
    axes[2].axhline(0,color='k',ls=':');axes[2].set(title='Paired risk difference ±2 MCSE',ylabel='MSE difference (ms²)')
    for ax in axes:
        ax.set(xlabel='True rho (noisier class)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'experiment.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':demo()
