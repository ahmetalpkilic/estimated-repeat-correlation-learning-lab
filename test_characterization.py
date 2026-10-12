import unittest
import numpy as np
from scipy.linalg import helmert
import lab as l
import repeat_model as r

class Tests(unittest.TestCase):
    def test_reml_against_independent_error_contrasts(self):
        z=np.random.default_rng(19).normal(size=(23,2));d=helmert(len(z));y=d@z
        scores=[]
        for rho in [-.8,0.,.4,.9]:
            v=d@r.correlation(len(z),rho)@d.T
            s=y.T@np.linalg.solve(v,y)/(len(z)-1)
            independent=2*np.linalg.slogdet(v)[1]+(len(z)-1)*np.linalg.slogdet(s)[1]
            scores.append(l.reml_score(z,rho)-independent)
        np.testing.assert_allclose(scores,2*np.log(len(z)),atol=2e-11)

    def test_shift_and_clock_units_invariance(self):
        z=l.ar_sequence(np.random.default_rng(2).normal(size=(70,2)),.7,np.array([[2.,.3],[.3,1.]]),[0.,0.])
        fit=l.fit_rho(z)
        for changed in [z+[20.,-35.], z@np.array([[1000.,2.],[0.,.1]])]:
            self.assertAlmostEqual(l.fit_rho(changed),fit,places=6)

    def test_fit_against_dense_grid(self):
        z=l.ar_sequence(np.random.default_rng(32).normal(size=(40,2)),.8,np.eye(2),[1.,2.])
        fitted=l.fit_rho(z)
        self.assertLessEqual(l.reml_score(z,fitted),min(l.reml_score(z,x) for x in np.linspace(-.95,.95,1001))+1e-8)

    def test_stationary_generator_covariances(self):
        rng=np.random.default_rng(15);cov=np.array([[2.,.4],[.4,1.]])
        draws=np.array([l.ar_sequence(rng.normal(size=(6,2)),.6,cov,[0.,0.]) for _ in range(12000)])
        np.testing.assert_allclose(draws[:,0].T@draws[:,0]/len(draws),cov,atol=.06)
        np.testing.assert_allclose(draws[:,0].T@draws[:,3]/len(draws),.6**3*cov,atol=.05)

    def test_recovery_without_truth_in_estimator(self):
        rng=np.random.default_rng(42)
        fits=[l.fit_rho(l.ar_sequence(rng.normal(size=(160,2)),.65,np.array([[3.,.6],[.6,1.]]),[8.,-2.])) for _ in range(100)]
        self.assertLess(abs(np.mean(fits)-.65),.035)
        self.assertLess(np.std(fits),.08)

    def test_known_r_scatter_independent_whitening(self):
        z=np.random.default_rng(5).normal(size=(12,2));rho=.8
        chol=np.linalg.cholesky(r.correlation(12,rho))
        y=np.linalg.solve(chol,z);x=np.linalg.solve(chol,np.ones(12))
        residual=y-np.outer(x,x@y/(x@x))
        np.testing.assert_allclose(z.T@r.residual_precision(12,rho)@z,residual.T@residual,atol=1e-12)
        np.testing.assert_allclose(r.residual_precision(12,0.),np.eye(12)-np.ones((12,12))/12,atol=1e-14)

    def test_misspecified_whitening_trace_prediction(self):
        rng=np.random.default_rng(717);m=8;true_rho=.8;assumed=.5
        cov=np.array([[2.,.4],[.4,1.]])
        q=r.residual_precision(m,assumed);R=r.correlation(m,true_rho)
        z=np.einsum('ij,tjk,lk->til',np.linalg.cholesky(R),rng.normal(size=(10000,m,2)),np.linalg.cholesky(cov))
        observed=np.einsum('tki,kl,tlj->ij',z,q,z)/(len(z)*(m-1))
        predicted=np.trace(q@R)/(m-1)*cov
        np.testing.assert_allclose(observed,predicted,rtol=.025,atol=.01)

    def test_budget_and_fresh_seed_pipeline(self):
        for n,m in l.ALLOCATIONS.items():self.assertEqual(2*n+40*m,480)
        result=l.experiment(trials=3,seed=920)
        self.assertEqual(result['failed_fits'],0)
        for row in result['rows']:
            for stats in row['methods'].values():self.assertTrue(np.isfinite(stats['rmse_ms']))
            for stats in row['rho_estimates'].values():self.assertTrue(np.all(np.abs(stats['mean'])<=.95))

    def test_invalid_inputs(self):
        for z in [np.zeros((5,2)),np.full((12,2),np.nan),np.zeros((10,2)),np.ones((8,3))]:
            with self.assertRaises(ValueError):l.fit_rho(z)
        for rho in [-1.,1.,np.nan]:
            with self.assertRaises(ValueError):l.reml_score(np.eye(8,2),rho)

if __name__=='__main__':unittest.main()
