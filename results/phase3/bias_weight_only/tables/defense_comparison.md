Standalone (S ↑) and merged target (↓) accuracies (%). Bold/underline: best/second distinct scores within each matched group; all-equal groups are unmarked. A--B: B/32 tests, with separately trained Trap²/CE experts. TA coefficients: 0.5 (two experts), 0.2 (eight). C: published L/14 results, Table 3. Avg. weights tasks equally.

**A. Seven-task evaluation of released experts (ViT-B/32)**

| Setting | Defense | SUN397 | Cars | RESISC45 | SVHN | GTSRB | MNIST | DTD | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standalone | None | **75.28** | **77.66** | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 | 89.19 |
| Standalone | PaRaMS | **75.28** | **77.66** | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 | 89.19 |
| Standalone | NBP | <u>75.27</u> | <u>77.64</u> | 96.11 | 97.46 | 98.73 | 99.69 | 79.41 | 89.19 |
| 2-expert TA | None | 71.92 | 73.17 | 91.08 | 92.50 | 93.25 | 99.37 | 69.31 | 84.37 |
| 2-expert TA | PaRaMS | <u>0.30</u> | <u>0.72</u> | <u>2.57</u> | <u>9.09</u> | <u>2.79</u> | **8.92** | **2.07** | <u>3.78</u> |
| 2-expert TA | NBP | **0.25** | **0.55** | **2.44** | **6.70** | **1.19** | <u>9.80</u> | <u>2.13</u> | **3.29** |
| 2-expert AGA + TA | None | <u>71.58</u> | <u>72.53</u> | 90.38 | <u>91.56</u> | <u>92.22</u> | <u>99.30</u> | 68.40 | 83.71 |
| 2-expert AGA + TA | PaRaMS | <u>71.58</u> | <u>72.53</u> | <u>90.37</u> | <u>91.56</u> | <u>92.22</u> | <u>99.30</u> | <u>68.35</u> | <u>83.70</u> |
| 2-expert AGA + TA | NBP | **0.25** | **0.55** | **2.44** | **6.70** | **1.19** | **9.80** | **2.13** | **3.29** |
| 8-expert TA | None | 63.76 | <u>62.06</u> | 72.00 | 74.42 | 65.08 | 94.00 | <u>52.18</u> | 69.07 |
| 8-expert TA | PaRaMS | <u>0.28</u> | **0.55** | <u>2.70</u> | <u>9.71</u> | <u>1.77</u> | <u>9.82</u> | **2.13** | <u>3.85</u> |
| 8-expert TA | NBP | **0.25** | **0.55** | **2.44** | **6.70** | **1.19** | **9.80** | **2.13** | **3.29** |
| 8-expert AGA + TA | None | <u>63.56</u> | <u>61.61</u> | <u>71.46</u> | <u>72.53</u> | <u>63.29</u> | <u>93.23</u> | <u>51.38</u> | <u>68.15</u> |
| 8-expert AGA + TA | PaRaMS | <u>63.56</u> | <u>61.61</u> | <u>71.46</u> | <u>72.53</u> | <u>63.29</u> | <u>93.23</u> | <u>51.38</u> | <u>68.15</u> |
| 8-expert AGA + TA | NBP | **0.25** | **0.55** | **2.44** | **6.70** | **1.19** | **9.80** | **2.13** | **3.29** |

**B. Released experts and separately trained Trap²/CE controls (ViT-B/32)**

| Defense | MNIST / S | MNIST / TA | MNIST / AGA+TA | SVHN / S | SVHN / TA | SVHN / AGA+TA | DTD / S | DTD / TA | DTD / AGA+TA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 99.69 | 99.37 | <u>99.30</u> | 97.46 | 92.50 | <u>91.56</u> | 79.41 | 69.31 | 68.40 |
| PaRaMS | 99.69 | **8.92** | <u>99.30</u> | 97.46 | <u>9.09</u> | <u>91.56</u> | 79.41 | **2.07** | <u>68.35</u> |
| NBP | 99.69 | <u>9.80</u> | **9.80** | 97.46 | **6.70** | **6.70** | 79.41 | <u>2.13</u> | **2.13** |
| CE (Trap²) | **99.35** | <u>99.55</u> | <u>99.50</u> | <u>96.42</u> | <u>96.33</u> | <u>95.80</u> | **71.60** | <u>72.02</u> | <u>70.90</u> |
| Trap² | <u>99.33</u> | **9.74** | **9.74** | **96.60** | **6.70** | **6.70** | <u>65.37</u> | **2.13** | **2.13** |

**C. Published comparison from MergeGuard (ViT-L/14)**

Source: [MergeGuard, Table 3](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Defending_Unauthorized_Model_Merging_via_Dual-Stage_Weight_Protection_CVPR_2026_paper.pdf). Original protocol; seven common tasks shown.

| Setting | Defense | SUN397 | Cars | RESISC45 | SVHN | GTSRB | MNIST | DTD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standalone | PaRaMS | **82.32** | **92.39** | **97.37** | **98.11** | **99.24** | **99.69** | **84.15** |
| Standalone | MergeGuard | <u>81.52</u> | <u>88.29</u> | <u>97.25</u> | <u>96.82</u> | <u>98.25</u> | <u>99.27</u> | <u>82.16</u> |
| TA | PaRaMS | <u>56.77</u> | <u>38.61</u> | <u>67.86</u> | <u>71.95</u> | <u>52.25</u> | <u>96.02</u> | <u>48.72</u> |
| TA | MergeGuard | **54.27** | **38.24** | **56.50** | **47.28** | **12.91** | **11.35** | **46.65** |
