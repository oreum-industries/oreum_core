# model.plot.py
# copyright 2021 Oreum OÜ
import arviz as az
import matplotlib.pyplot as plt
import patsy as pt


def facetplot_azid_dist(azid, rvs, rvs_hack_extra=0, group='posterior', ref_vals=None):
    """Convenience: plot Krushke style in facets """
    # TODO unpack the compressed rvs from the azid
    
    # m, n = 2, (len(rvs) // 2) + (len(rvs) % 2)
    m, n = 2, ((len(rvs)+rvs_hack_extra) // 2) + ((len(rvs)+rvs_hack_extra) % 2)
    f, ax1d = plt.subplots(n, m, figsize=(m*6, 2.2*n))
    kw = {}
    if ref_vals is not None:
        kw['ref_vals'] = ref_vals
    _ = az.plot_posterior(azid, group=group, ax=ax1d, var_names=rvs, **kw)
    f.suptitle(group, y=0.9 + n*0.005)
    f.tight_layout()

