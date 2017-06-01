import numpy as np

to_know = """
Some definitions:
R.P - Region Proper - A linear region containing multiple instances, most of the same type
(this is a construct which we hypothesize the network makes use of)

I guess the big question is do networks classify the same on a region (proper) or is there a
significant slope that classfies on the degree of that slope. The problem with this question is that
in the case of the region proper this comes down to its definition. What we need to be asking is,
are classifications being changed due to a movement from some a > 0 to a' > 0. Or by a > 0 simply
dissipearing? If the latter is true we are ripe for using region analysis.

The following would be practical to know:

1) Same region same prediction? - If, for those regions which have more than one instance, do they share predictions/actuals
of different types, for what percentage of these is this the case (could get this done with the
histogram you already have)

Why - If they mostly have the same prediction this COULD lead us to suspect that a network does classify
instances in the same region proper as the same. I say COULD because the actual regions could be so
small that they only contain near identical instances, where the broader region proper is in fact
capable of using the slope to classify different regions.

2) Different super region, different prediction? - Again, would be much better if we had some way
of obtaining the region proper. Are regions with the same t at layer 2, but different t's at layer
1, able to be distinguished when required. Check this for correct predictions based on different
instances, across all instances (with correct predictions) sharing the same t but different super regions.
Compute the percetage of similar predictions, is it close to 10% i.e all are distinguished or close
to 100% all are similar. I suspect this one is less susceptible to bias from the exact region.

Why - What I would really like to know is are the EOS's of two regions in case 2) substantially
different. I guess they would have to be, (in both the proper and exact cases), if different
predictions commonly resulted.

3) K Nearest Region Neighbour - For incorrect predictions, find the nearest network neighbours
see if these guys can allow us to predict when a network is going to fuck up. It could be all
due to some rouge C.U.N.T of an output

Why - Going for the jugular, if it worked it would be a pratical way to approach uncertainty
in NNs.

The original theory is that for the most part, the network identifies single linear regions, where
each linear region only contains the same predictions/instances. Suppose this is the case but there are
one or more trivial activations, which do not / are not supposed to influence the output. Less
those trivial activations, we have the proper region.

If we can distinguish these proper regions, and the theory is true, then we should be able to tell:
1) When the instance is in a proper region, but has been bashed by a bad activation
2) When the instance is outside, or in multiple linear regions, in which case we note the uncertainty

The first order of business is identifying these trivial activations, how can we do it?



If the

If you can define your question formally,  it is certain an algorithm exist to find it. Whether
or not it is computationally tractable is another question.

"""

def k_nearest(f1, f2):
    ts1 = f1.all_final_ts()
    ts2 = f1.all_final_ts()
    for ts1