# Some Related Papers for Building Responsible Foundations for Robust Pre-training

---

## Current DNN variance and robustness issues (and countermeasures)

1. Covariate shift
  
  - [Overparameterization Improves Robustness to Covariate Shift in High Dimensions (neurips.cc)](https://proceedings.neurips.cc/paper/2021/hash/73fed7fd472e502d8908794430511f4d-Abstract.html)
    
  - [Improving robustness against common corruptions by covariate shift adaptation (neurips.cc)](https://proceedings.neurips.cc/paper/2020/hash/85690f81aadc1749175c187784afc9ee-Abstract.html)
    
2. Vulerable to adversarial examples or noise
  

- [How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness? (neurips.cc)](https://proceedings.neurips.cc/paper/2021/hash/22b1f2e0983160db6f7bb9f62f4dbb39-Abstract.html)
  
- [Robust Pre-Training by Adversarial Contrastive Learning (neurips.cc)](https://proceedings.neurips.cc/paper/2020/hash/ba7e36c43aff315c00ec2b8625e3b719-Abstract.html)
  
- [Adversarial Robustness: From Self-Supervised Pre-Training to Fine-Tuning | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9156433)
  
- [Fuzz testing based data augmentation to improve robustness of deep neural networks | Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering](https://dl.acm.org/doi/10.1145/3377811.3380415)
  
- [Renofeation: A Simple Transfer Learning Method for Improved Adversarial Robustness | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9523126)
  
- [Learn2Perturb: An End-to-End Feature Perturbation Learning to Improve Adversarial Robustness (thecvf.com)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jeddi_Learn2Perturb_An_End-to-End_Feature_Perturbation_Learning_to_Improve_Adversarial_Robustness_CVPR_2020_paper.pdf)
  

**Regularization-related (reweight the less confident data points or regularization loss)**

- [Improved Regularization and Robustness for Fine-tuning in Neural Networks (neurips.cc)](https://proceedings.neurips.cc/paper/2021/hash/e4a93f0332b2519177ed55741ea4e5e7-Abstract.html)
  
- [Adversarial Robustness via Fisher-Rao Regularization | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9773978)
  
- [On Connections Between Regularizations for Improving DNN Robustness | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9133291)
  

3. HPO for robustness:
  
  - [Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with Dragonfly](https://jmlr.org/papers/v21/18-223.html)
    
    **The non-robustness described in this tool**: Conventional BO methods tend to be sensitive to the choice of the acquisition and the parameters of the underlying Bayesian model.
    
  - [DEHB: Evolutionary Hyberband for Scalable, Robust and Efficient Hyperparameter Optimization | IJCAI](https://www.ijcai.org/proceedings/2021/296)
    
    **Problem**: High-dimensional optimziation problems with discrete input dimensions.
    
    **Solution**: Combine the advantages of the popular bandit-based HPO method Hyperband (HB) and the evolutionary search approach of Differential Evolution (DE) to yield a new HPO method which named DEHB.
    
  - [Exploring the Hyperparameter Landscape of Adversarial Robustness](https://arxiv.org/abs/1905.03837)
    
    **Contributions**: 1) It is difficult to find effective adversarial training parameter settings that hit a sweet spot in the surface plots: maximizing robustness while minimizing the reduction in accuracy. 2) HPO strategies to balance these two terms.
    
  
  ---
  

## DNN Robustness-Related

1. [Improved Regularization and Robustness for Fine-tuning in Neural Networks (neurips.cc)](https://proceedings.neurips.cc/paper/2021/hash/e4a93f0332b2519177ed55741ea4e5e7-Abstract.html)
  **Problem**: Vulnerable to noise when the size of the target data set is much smalller than the capacity of the pre-trained model.
  
  **Solution**: Propose regularized self-labeling---the interpolation between regularization and self-labeling methods (maybe not the work of HPO?)
  
2. [An Adaptive Robustness Evolution Algorithm With Self-Competition and its 3D Deployment for Internet of Things | IEEE/ACM Transactions on Networking](https://dl.acm.org/doi/10.1109/TNET.2021.3113916)
   **Problems**: the optimization of the neural networks is vulnerable to  malicious attacks
  
  **Solution**: Heuristic algorithms, particularly genetic algorithms,  can effectively cope with such problems.
  
3. [Benchmarking Adversarial Robustness on Image Classification | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9157625)
  
  **Problem**: Adversarial robustness on image classification tasks
   **Main Work**: Establish a comprehensive, rigorous, and coherent benchmark
  
  **Conclusion**:  1) The robustness between models can change across different attack configurations; 2) As one of the most effective defense techniques, adversarial training can generalize across different threat models; 3) Randomization-based defenses are more robust to query-based black-box attacks.
  
  **The possible useful part for the HPO-enhanced robust solution**: Maybe Quantitative certification of the robustness of an underlying pre-trained model?
  
4. [Adversarial Robustness: From Self-Supervised Pre-Training to Fine-Tuning | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9156433)
  
  **Main Work**: 1) Introduce adversarial training into self-supervision, and provide general-purpose robust pretrained models,  2) Ensemble several pretraining tasks, which boosts robustness more.
  
  **The possible useful part for the HPO-enhanced robust solution**: Add adversarial training into the HPO process? How to add? ~~Manual craft the adversarial parameters like adversarial samples.(Out of the scope of black-box attack)~~
  
5. [Towards Verifying Robustness of Neural Networks Against A Family of Semantic Perturbations | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9157375)
  
  **Main Work**: proposed semantic perturbation layers (SP-layers) to the input layer of any given model, and then any ℓp-norm based verification tools can be used to verify the model robustness against semantic perturbations.
  
  **The possible useful part for the HPO-enhanced robust solution**:  Quantitative certification of the robustness (to validate the robustness of the pre-trained model on semantic perturbations?)
  
6. [Adversarial Robustness via Fisher-Rao Regularization | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9773978)
  
  **Main work**: Enhance the categorical cross entropy with the proposed Fisher-Rao Regularization
  
  **Results or Conclusions**: showing up to a simultaneous 1% of improvement in terms of clean and robust performances while reducing the training time by 20% over the best-performing methods.
  
  **The possible useful part for the HPO-enhanced robust solution**: Though HPO is black-box optimization, the loss is visible for HPO, then maybe this extra term can be added into the loss for the robustness performance?
  
7. [On Connections Between Regularizations for Improving DNN Robustness | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/9133291)
  
  **Main work:** Enhance the DNNs with Regularizations (input-gradient regularization, Jacobian regularization, curvature regularization, and a cross-Lipschitz functional), maybe the combination with loss function?
  
  **The possible useful part for the HPO-enhanced robust solution**: Same with paper
  
8. [Fuzz testing based data augmentation to improve robustness of deep neural networks | Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering](https://dl.acm.org/doi/10.1145/3377811.3380415)
  
  **Main work:** Mutation-based fuzzing to augment the training data of DNNs, with the objective of enhancing their robustness. Cast the DNN data augmentation problem as an optimization problem. It uses genetic search to generate the most suitable variant of an input data to use for training the DNN, while simultaneously identifying opportunities to accelerate training by skipping augmentation in many instances.
  
  **The possible useful part for the HPO-enhanced robust solution**: Data augment techniques?
  
9. [Importance-driven deep learning system testing | Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering](https://dl.acm.org/doi/10.1145/3377811.3380391)
  
  **Main work**: Propose a systematic testing methodology accompanied by an Importance-Driven (IDC) test adequacy criterion for DL systems
  
  **The possible useful part for the HPO-enhanced robust solution**: Quantitative certification of the robustness (to validate the robustness of the pre-trained model based on the importance-driven criterion?)
  
10. [Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with Dragonfly](https://jmlr.org/papers/v21/18-223.html)
  
  **The non-robustness described in this tool**: Conventional BO methods tend to be sensitive to the choice of the acquisition and the parameters of the underlying Bayesian model.
  
  **The work to address the sensitivity problem:** We describe new randomised approaches implemented in Dragonfly which stochastically sample among an available set of acquisition choices and model parameters instead of relying on a single value for the entire optimisation routine.
  
  **The possible useful part for HPO-enhanced robust solution:** How the randomised approaches occur in the Droagonfly? Is this process can be optimized via a steady process? Does the unsteady exists in this tool?
  
11. [You Shouldn’t Trust Me: Learning Models Which Conceal Unfairness From Multiple Explanation Methods](http://ceur-ws.org/Vol-2560/paper8.pdf) (Safe AI@AAAI 2020)
  
  **Main work**: Introduce a practical approach to modify an existing model and downgrade the apparent importance of a sensitive feature according to explanation methods.
  
  **The possible useful part for HPO-enhanced robust solution:** Can HPO realize a algorithm to address this or this kind of attack?
  
12. [How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness? (neurips.cc)](https://proceedings.neurips.cc/paper/2021/hash/22b1f2e0983160db6f7bb9f62f4dbb39-Abstract.html)
  
  **Problem**: The fine-tuning of pre-trained language models is strikingly vulnerable to adversarial examples. Adversarial training, the prevalent defense technique, does not directly fit a conventional fine-tuning scenario, because it suffers severely from catastrophic forgetting: failing to retain the generic and robust linguistic features that have already been captured by the pre-trained model.
  
  **Main work**: Propose Robust Informative Fine-Tuning (RIFT), a novel adversarial fine-tuning method from an information-theoretical perspective to retain the features learned from the pre-trained model throughout the entire fine-tuning process rather than just the pre-trained weights for initialization to maintain the robustness.
  
13. [Renofeation: A Simple Transfer Learning Method for Improved Adversarial Robustness | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9523126)
  
  **Main work**: Considering adversarial attack on the last layer of fine-tuning models will deceive the trained models, this work proposes a noise-feature distillation method to train a model with random initialization for the robustness.  How the noise feature distillation works needs a further study.
  
  **The possible useful part for the HPO-enhanced robust solution**: UNKNOWN
  
14. [Learn2Perturb: An End-to-End Feature Perturbation Learning to Improve Adversarial Robustness (thecvf.com)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jeddi_Learn2Perturb_An_End-to-End_Feature_Perturbation_Learning_to_Improve_Adversarial_Robustness_CVPR_2020_paper.pdf)
  
  **Main Work**:  The proposed Perturbation-injection modules are incorporated at each layer to perturb the feature space and increase uncertainty in the network. This attack requires the model is a white-box model, which is not suitable for the HPO.
  

---

## PTM-related

PTMs is vulnerable to adversarial attacks

Ref. [209] successfully attacked the fine-tuned BERT on text classification and textual entailment with adversarial examples. Ref. [210] defined universal adversarial triggers that can induce a model to produce a specific-purpose prediction when concatenated to any input. Some triggers can even cause the GPT-2 model to generate racist text. Ref. [211] showed BERT is not robust on misspellings. (From [Pre-trained models for natural language processing: A survey](https://link.springer.com/article/10.1007/s11431-020-1647-3))