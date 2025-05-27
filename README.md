# MLT new approach

## Short summary of the algorithm in [Arora et. al.](https://www.sciencedirect.com/science/article/pii/S0167865507002905)

1. Recursively repeat steps 2–6 for $\frac{n}{2}-1$ times; 
	where n is the number of thresholds.
2. For range $R = [a, b]$ ; initially a = 0 and b = 255.
3. Find mean $(\mu)$ and standard deviation $(\sigma)$ of intensity distribution of the pixels in **R**.
4. Sub-ranges’ boundaries $T_1$ and $T_2$ are calculated as $T_1 = \mu - \kappa_1 \cdot \sigma$ and $T_2 = \mu + \kappa_2 \cdot \sigma$; where $\kappa_1$ and $\kappa_2$ are free parameters.
5. Pixels with intensity values in the interval $[a,T_1]$ and $[T_2,b]$ are assigned threshold values equal to the respective weighted means of their values.
6. we assign new $a$ & $b$ as: $a= T_1 + 1$, $b= T_2 - 1$.
7. Finally after iterating over above process we need to assign for the pixels of remaining pixels in the range $(T_1,T_2)$
	repeat step 5 with $T_1  = \mu$ and with $T_2 = \mu + 1$. 

>  On the other hand, choosing the weighted mean of a class as the replacement value ensures that intra-class variance of sub-ranges is minimum leading to increased PSNR and quality of image

[[PATREC_4248.pdf#page=4&selection=16,42,28,5|PATREC_4248, page 4]]

> 
[[PATREC_4248.pdf#page=4&selection=218,0,218,9|PATREC_4248, page 4]]
## Comparing Mean vs mode using

- PSNR
- SSIM (**structural similarity index measure**)
- Entropy measure

## Writing own pipeline for the same.

- [ ] Above algorithm in python
- [ ] Plotting
  - [ ] PSNR
  - [ ] Distribution with thresholds
  - [ ] 