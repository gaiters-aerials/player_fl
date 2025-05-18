# Results

Below we present results for layer analytics and evaluation. We introduce metric definitions before presenting the results.

## Layer Analytics: Comparing models trained on non-IID data

### Metric definitions

#### Gradient variance  
Gradient variance, proposed by \cite{jiang2019fantastic}, is defined as:  

<img src="https://latex.codecogs.com/svg.latex?\text{Var}(\nabla\theta_i):=\frac{1}{n}\sum_{j=1}^n\left(\nabla\theta_i^j-\overline{\nabla\theta_i}\right)^T\left(\nabla\theta_i^j-\overline{\nabla\theta_i}\right)" />

where $\theta_i^{\,j}$ is parameter *j* in layer *i*,  
$\nabla\theta_i^{\,j}$ its gradient, and $\overline{\nabla\theta_i}$ the mean gradient of all parameters in that layer.

#### Hessian eigenvalue sum  
Proposed by \cite{chaudhari2019entropy}. Each entry in the Hessian for layer *i* is  

<img src="https://latex.codecogs.com/svg.latex?(H_i)_{jk}=\frac{\partial^2\mathcal{L}}{\partial\theta_i^j\partial\theta_i^k}" />

The sum of its eigenvalues is then  

<img src="https://latex.codecogs.com/svg.latex?\text{Tr}(H_i)=\sum_{p=1}^n\lambda_i^p" />


#### Sample-representation similarity (CKA)  
Using Centered Kernel Alignment \cite{kornblith2019similarity}:  

<img src="https://latex.codecogs.com/svg.latex?CKA(X_i,Y_i)=\frac{||Y_i^TX_i||_F^2}{||X_i^TX_i||_F^2||Y_i^TY_i||_F^2}" />

where $X_i$ and $Y_i$ are the representations after layer *i* from two distinct models.



### Gradient variance

Figures \ref{sfig:grad_var_first}, \ref{sfig:grad_var_best}, and \ref{sfig:grad_var_best_fl} display the gradient variance across all datasets, corresponding to the models trained for a single epoch, final model after independent training, and final model after FL training, respectively.
Layer Gradient Variance
After One Epoch
<table>
  <tr align="center">
    <td width="25%"><img src="figures/path_to_image1.png" width="100%"><br><em>(A) DatasetName1</em></td>
    <td width="25%"><img src="figures/path_to_image2.png" width="100%"><br><em>(B) DatasetName2</em></td>
    <td width="25%"><img src="figures/path_to_image3.png" width="100%"><br><em>(C) DatasetName3</em></td>
    <td width="25%"><img src="figures/path_to_image4.png" width="100%"><br><em>(D) DatasetName4</em></td>
  </tr>
  <tr align="center">
    <td width="25%"><img src="figures/path_to_image5.png" width="100%"><br><em>(E) DatasetName5</em></td>
    <td width="25%"><img src="figures/path_to_image6.png" width="100%"><br><em>(F) DatasetName6</em></td>
    <td width="25%"><img src="figures/path_to_image7.png" width="100%"><br><em>(G) DatasetName7</em></td>
    <td width="25%"></td>
  </tr>
</table>
Figure X: Layer gradient variance after one epoch. All models identically initialized and independently trained on non-IID data.
Final Models (Independent Training)
<table>
  <tr align="center">
    <td width="25%"><img src="figures/FMNIST_Gradient_Variance_best-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="25%"><img src="figures/EMNIST_Gradient_Variance_best-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="25%"><img src="figures/CIFAR_Gradient_Variance_best-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
    <td width="25%"><img src="figures/ISIC_Gradient_Variance_best-1.png" width="100%"><br><em>(D) ISIC-2019</em></td>
  </tr>
  <tr align="center">
    <td width="25%"><img src="figures/Sentiment_Gradient_Variance_best-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="25%"><img src="figures/mimic_Gradient_Variance_best-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="25%"><img src="figures/Heart_Gradient_Variance_best-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
    <td width="25%"></td>
  </tr>
</table>
Figure: Layer gradient variance for final models. All models identically initialized and independently trained on non-IID data.
Final Models (FL Training)
<table>
  <tr align="center">
    <td width="25%"><img src="figures/FMNIST_Gradient_Variance_best_federated-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="25%"><img src="figures/EMNIST_Gradient_Variance_best_federated-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="25%"><img src="figures/CIFAR_Gradient_Variance_best_federated-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
    <td width="25%"><img src="figures/ISIC_Gradient_Variance_best_federated-1.png" width="100%"><br><em>(D) ISIC-2019</em></td>
  </tr>
  <tr align="center">
    <td width="25%"><img src="figures/Sentiment_Gradient_Variance_best_federated-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="25%"><img src="figures/mimic_Gradient_Variance_best_federated-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="25%"><img src="figures/Heart_Gradient_Variance_best_federated-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
    <td width="25%"></td>
  </tr>
</table>
Figure: Layer gradient variance for final models. Models trained via FL on non-IID data.
Hessian Eigenvalue Sum
Figures \ref{sfig:hess_eig_first}, \ref{sfig:hess_eig_best}, and \ref{sfig:hess_eig_best_fl} display the hessian eigenvalue sum across all datasets, corresponding to the models trained for a single epoch, final model after independent training, and final model after FL training, respectively.
After One Epoch
<table>
  <tr align="center">
    <td width="25%"><img src="figures/FMNIST_Hessian_EV_sum_first-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="25%"><img src="figures/EMNIST_Hessian_EV_sum_first-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="25%"><img src="figures/CIFAR_Hessian_EV_sum_first-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
    <td width="25%"><img src="figures/ISIC_Hessian_EV_sum_first-1.png" width="100%"><br><em>(D) ISIC-2019</em></td>
  </tr>
  <tr align="center">
    <td width="25%"><img src="figures/Sentiment_Hessian_EV_sum_first-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="25%"><img src="figures/mimic_Hessian_EV_sum_first-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="25%"><img src="figures/Heart_Hessian_EV_sum_first-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
    <td width="25%"></td>
  </tr>
</table>
Figure: Layer hessian eigenvalue sum after one epoch. All models identically initialized and independently trained on non-IID data.
Final Models (Independent Training)
<table>
  <tr align="center">
    <td width="25%"><img src="figures/FMNIST_Hessian_EV_sum_best-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="25%"><img src="figures/EMNIST_Hessian_EV_sum_best-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="25%"><img src="figures/CIFAR_Hessian_EV_sum_best-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
    <td width="25%"><img src="figures/ISIC_Hessian_EV_sum_best-1.png" width="100%"><br><em>(D) ISIC-2019</em></td>
  </tr>
  <tr align="center">
    <td width="25%"><img src="figures/Sentiment_Hessian_EV_sum_best-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="25%"><img src="figures/mimic_Hessian_EV_sum_best-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="25%"><img src="figures/Heart_Hessian_EV_sum_best-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
    <td width="25%"></td>
  </tr>
</table>
Figure: Layer hessian eigenvalue sum for the final models. All models identically initialized and independently trained on non-IID data.
Final Models (FL Training)
<table>
  <tr align="center">
    <td width="25%"><img src="figures/FMNIST_Hessian_EV_sum_best_federated-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="25%"><img src="figures/EMNIST_Hessian_EV_sum_best_federated-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="25%"><img src="figures/CIFAR_Hessian_EV_sum_best_federated-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
    <td width="25%"><img src="figures/ISIC_Hessian_EV_sum_best_federated-1.png" width="100%"><br><em>(D) ISIC-2019</em></td>
  </tr>
  <tr align="center">
    <td width="25%"><img src="figures/Sentiment_Hessian_EV_sum_best_federated-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="25%"><img src="figures/mimic_Hessian_EV_sum_best_federated-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="25%"><img src="figures/Heart_Hessian_EV_sum_best_federated-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
    <td width="25%"></td>
  </tr>
</table>
Figure: Layer hessian eigenvalue sum for the final models. Models trained via FL on non-IID data.
Sample Representation
Figures \ref{sfig:sample_rep_first} and \ref{sfig:sample_rep_best} display the sample representation across all datasets, corresponding to the models trained for a single epoch and the final models, respectively.
After One Epoch
<table>
  <tr align="center">
    <td width="33.3%"><img src="figures/FMNIST_similarity_first-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="33.3%"><img src="figures/EMNIST_similarity_first-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="33.3%"><img src="figures/CIFAR_similarity_first-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
  </tr>
  <tr align="center">
    <td width="33.3%"><img src="figures/Sentiment_similarity_first-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="33.3%"><img src="figures/mimic_similarity_first-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="33.3%"><img src="figures/Heart_similarity_first-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
  </tr>
</table>
Figure: Layer sample representation similarity after one epoch. All models identically initialized and independently trained on non-IID data.
Final Models
<table>
  <tr align="center">
    <td width="33.3%"><img src="figures/FMNIST_similarity_best-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="33.3%"><img src="figures/EMNIST_similarity_best-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="33.3%"><img src="figures/CIFAR_similarity_best-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
  </tr>
  <tr align="center">
    <td width="33.3%"><img src="figures/Sentiment_similarity_best-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="33.3%"><img src="figures/mimic_similarity_best-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="33.3%"><img src="figures/Heart_similarity_best-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
  </tr>
</table>
Figure: Layer sample representation similarity for final models. All models identically initialized and independently trained on non-IID data.
Federated Sensitivity
Figures \ref{sfig:layer_imp_best} and \ref{sfig:layer_imp_best_fl} display the federated sensitivity score across all datasets for the final model after independent training, and final model after FL training, respectively.
Final Models (Independent Training)
<table>
  <tr align="center">
    <td width="25%"><img src="figures/FMNIST_Layer_importance_best-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="25%"><img src="figures/EMNIST_Layer_importance_best-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="25%"><img src="figures/CIFAR_Layer_importance_best-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
    <td width="25%"><img src="figures/ISIC_Layer_importance_best-1.png" width="100%"><br><em>(D) ISIC-2019</em></td>
  </tr>
  <tr align="center">
    <td width="25%"><img src="figures/Sentiment_Layer_importance_best-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="25%"><img src="figures/mimic_Layer_importance_best-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="25%"><img src="figures/Heart_Layer_importance_best-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
    <td width="25%"></td>
  </tr>
</table>
Figure: Federation sensitivity for final models. All models identically initialized and independently trained on non-IID data.
Final Models (FL Training)
<table>
  <tr align="center">
    <td width="25%"><img src="figures/FMNIST_Layer_importance_best_federated-1.png" width="100%"><br><em>(A) FashionMNIST</em></td>
    <td width="25%"><img src="figures/EMNIST_Layer_importance_best_federated-1.png" width="100%"><br><em>(B) EMNIST</em></td>
    <td width="25%"><img src="figures/CIFAR_Layer_importance_best_federated-1.png" width="100%"><br><em>(C) CIFAR-10</em></td>
    <td width="25%"><img src="figures/ISIC_Layer_importance_best_federated-1.png" width="100%"><br><em>(D) ISIC-2019</em></td>
  </tr>
  <tr align="center">
    <td width="25%"><img src="figures/Sentiment_Layer_importance_best_federated-1.png" width="100%"><br><em>(E) Sent-140</em></td>
    <td width="25%"><img src="figures/mimic_Layer_importance_best_federated-1.png" width="100%"><br><em>(F) MIMIC-III</em></td>
    <td width="25%"><img src="figures/Heart_Layer_importance_best_federated-1.png" width="100%"><br><em>(G) Fed-Heart-Disease</em></td>
    <td width="25%"></td>
  </tr>
</table>
Figure: Federation sensitivity for final models. Models trained via FL on non-IID data.

## Results

### Model training

Table: Learning rates used

| Algorithm         | FMNIST           | EMNIST           | CIFAR            | ISIC             | Heart            | Sent-140         | MIMIC-III        |
| :---------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- |
| Local client      | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-2}$ | $5\cdot 10^{-4}$ | $8\cdot 10^{-5}$ |
| FedAvg            | $5\cdot 10^{-4}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-1}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ |
| FedProx           | $5\cdot 10^{-4}$ | $5\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-2}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ |
| pFedMe            | $5\cdot 10^{-2}$ | $5\cdot 10^{-2}$ | $5\cdot 10^{-2}$ | $5\cdot 10^{-3}$ | $1\cdot 10^{-1}$ | $1\cdot 10^{-2}$ | $1\cdot 10^{-3}$ |
| Ditto             | $5\cdot 10^{-4}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-2}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ |
| LocalAdaptation   | $5\cdot 10^{-4}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-2}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ |
| BABU              | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-1}$ | $8\cdot 10^{-5}$ | $5\cdot 10^{-4}$ |
| PLayer-FL         | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ | $1\cdot 10^{-2}$ | $8\cdot 10^{-5}$ | $3\cdot 10^{-4}$ |
| PLayer-FL-1       | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ | $5\cdot 10^{-2}$ | $8\cdot 10^{-5}$ | $8\cdot 10^{-5}$ |
| PLayer-FL+1       | $1\cdot 10^{-3}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-4}$ | $1\cdot 10^{-3}$ | $5\cdot 10^{-2}$ | $1\cdot 10^{-4}$ | $5\cdot 10^{-4}$ |

Table: Learning rate grid, loss function, number of epochs and number of runs for each dataset

| Dataset    | Learning rate grid                           | Loss             | Epochs | Runs |
| :--------- | :--------------------------------------------- | :--------------- | :----- | :--- |
| FashionMNIST | $1\cdot10^{-3}$, $5\cdot10^{-4}$, $1\cdot10^{-4}$, $8\cdot10^{-5}$ | Cross Entropy    | 75     | 10   |
| EMNIST     | $5\cdot10^{-3}$, $1\cdot10^{-3}$, $5\cdot10^{-4}$, $1\cdot10^{-4}$, $8\cdot10^{-5}$ | Cross Entropy    | 75     | 10   |
| CIFAR-10   | $5\cdot10^{-3}$, $1\cdot10^{-3}$, $5\cdot10^{-4}$, $1\cdot10^{-4}$ | Cross Entropy    | 50     | 10   |
| ISIC-2019  | $1\cdot10^{-3}$, $5\cdot10^{-3}$, $1\cdot10^{-4}$ | Multiclass Focal | 50     | 3    |
| Sent-140   | $1\cdot10^{-3}$, $5\cdot10^{-4}$, $1\cdot10^{-4}$, $8\cdot10^{-5}$ | Cross Entropy    | 75     | 20   |
| Heart      | $5\cdot10^{-1}$, $1\cdot10^{-1}$, $5\cdot10^{-2}$, $1\cdot10^{-2}$, $5\cdot10^{-3}$ | Multiclass Focal | 50     | 50   |
| MIMIC-III  | $5\cdot10^{-4}$, $1\cdot10^{-4}$, $3\cdot10^{-4}$, $8\cdot10^{-5}$ | Multiclass Focal | 50     | 10   |

Table \ref{supp:lr} presents the learning rates utilized for each algorithm and Table \ref{supp:hyperparams} presents the learning rate grid explored, in addition to the loss function, the number of training epochs, and the count of independent training runs. With the exception of pFedME, the AdamW optimizer was used for all algorithms. For pFedME, we adopted the specific optimizer presented by the original authors, which integrates Moreau envelopes into the training process \cite{t2020pfedme}. To account for this, we multiply the learning rate grid tested by 100 as early testing showed the pFedME optimizer demonstrated improved performance with higher learning rates. All experiments were ran on 1 Tesla V100 16GB node.

### Metric definitions

#### Macro-averaged F1 score  

<img src="https://latex.codecogs.com/svg.latex?\text{F1}_{\text{macro}}=\frac{1}{N}\sum_{i=1}^N2\cdot\frac{\text{precision}_i\cdot\text{recall}_i}{\text{precision}_i+\text{recall}_i}" />

with *N* classes; useful when classes are imbalanced.


#### Algorithm fairness  
Variance of per-client performance \cite{divi2021new}:  

<img src="https://latex.codecogs.com/svg.latex?\text{Fairness}=\frac{1}{C}\sum_{c=1}^C\left(P_c-\overline{P_c}\right)^2" />


#### Algorithm incentivization  
Percentage of clients whose personalized model beats both their local-site model and the global FedAvg model \cite{cho2022federate}:  

<img src="https://latex.codecogs.com/svg.latex?\text{Incentivization}=\frac{1}{C}\sum_{c=1}^C\mathbb{I}\{P_c>\max(S_c,G_c)\}" />


### F1 score: fairness and incentive

Table: Variance in clients' F1 score (fairness). In **bold** is fairest model. Friedman rank test p-value $<5\times10^{-3}$

| Algorithm          | FMNIST          | EMNIST          | CIFAR           | ISIC            | Heart           | Sent-140        | MIMIC-III       | Rank |
| :----------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :--- |
| Fedprox            | $3.0\cdot 10^{-4}$ | $8.0\cdot 10^{-4}$ | $4.0\cdot 10^{-4}$ | $1.1\cdot 10^{-2}$ | $9.0\cdot 10^{-2}$ | $4.1\cdot 10^{-3}$ | $1.7\cdot 10^{-3}$ | 7.7  |
| pFedMe             | $3.0\cdot 10^{-4}$ | $6.0\cdot 10^{-4}$ | $5.0\cdot 10^{-4}$ | $4.2\cdot 10^{-3}$ | $\mathbf{7.1\cdot 10^{-2}}$ | $3.7\cdot 10^{-2}$ | $\mathbf{1.4\cdot 10^{-3}}$ | 4.3  |
| Ditto              | $5.0\cdot 10^{-4}$ | $6.0\cdot 10^{-4}$ | $2.0\cdot 10^{-4}$ | $3.6\cdot 10^{-3}$ | $8.2\cdot 10^{-2}$ | $3.6\cdot 10^{-2}$ | $1.6\cdot 10^{-3}$ | 5.0  |
| LocalAdaptation    | $\mathbf{1.0\cdot 10^{-4}}$ | $7.0\cdot 10^{-4}$ | $4.0\cdot 10^{-4}$ | $9.0\cdot 10^{-3}$ | $8.3\cdot 10^{-2}$ | $4.1\cdot 10^{-2}$ | $2.2\cdot 10^{-3}$ | 6.4  |
| FedBABU            | $2.0\cdot 10^{-4}$ | $4.0\cdot 10^{-4}$ | $4.0\cdot 10^{-4}$ | $2.1\cdot 10^{-3}$ | $8.4\cdot 10^{-2}$ | $3.5\cdot 10^{-2}$ | $2.0\cdot 10^{-3}$ | 4.4  |
| FedLP              | $3.0\cdot 10^{-4}$ | $4.0\cdot 10^{-4}$ | $1.0\cdot 10^{-3}$ | $1.1\cdot 10^{-2}$ | $8.5\cdot 10^{-2}$ | $3.7\cdot 10^{-2}$ | $1.6\cdot 10^{-3}$ | 6.1  |
| FedLAMA            | $2.0\cdot 10^{-4}$ | $6.0\cdot 10^{-4}$ | $5.0\cdot 10^{-4}$ | $4.7\cdot 10^{-3}$ | $8.7\cdot 10^{-2}$ | $4.1\cdot 10^{-2}$ | $2.0\cdot 10^{-3}$ | 5.6  |
| pFedLA             | $4.0\cdot 10^{-4}$ | $1.0\cdot 10^{-4}$ | $\mathbf{1.0\cdot 10^{-4}}$ | $3.5\cdot 10^{-3}$ | $7.4\cdot 10^{-2}$ | $4.1\cdot 10^{-2}$ | $2.1\cdot 10^{-3}$ | 5.1  |
| PLayer-FL          | $4.0\cdot 10^{-4}$ | $5.0\cdot 10^{-4}$ | $6.0\cdot 10^{-4}$ | $1.3\cdot 10^{-3}$ | $7.3\cdot 10^{-2}$ | $\mathbf{3.3\cdot 10^{-2}}$ | $1.5\cdot 10^{-3}$ | **3.8** |
| PLayer-FL-Random | $8.0\cdot 10^{-4}$ | $\mathbf{3.0\cdot 10^{-4}}$ | $1.0\cdot 10^{-3}$ | $\mathbf{7.0\cdot 10^{-4}}$ | $7.3\cdot 10^{-2}$ | $3.4\cdot 10^{-2}$ | $1.9\cdot 10^{-3}$ | 6.5  |

Table: Incentivized participation rate (%) using F1 score. In **bold** is model with highest IPR. Friedman rank test p-value $=0.043$

| Algorithm          | FMNIST | EMNIST | CIFAR | ISIC | Heart | Sentiment | Mimic-III | Rank |
| :----------------- | :----- | :----- | :---- | :--- | :---- | :-------- | :-------- | :--- |
| FedProx            | 0.0    | 20.0   | 40.0  | 0.0  | 0.0   | 6.7       | 50.0      | 5.4  |
| pFedMe             | 80.0   | 20.0   | 0.0   | 0.0  | **50.0** | 6.7       | 0.0       | 5.1  |
| Ditto              | 60.0   | 0.0    | 0.0   | 0.0  | 0.0   | **26.7** | 25.0      | 5.6  |
| LocalAdaptation    | 0.0    | 40.0   | 40.0  | 0.0  | 0.0   | 0.0       | **75.0** | 5.2  |
| FedBABU            | 0.0    | 40.0   | 80.0  | 0.0  | 0.0   | 20.0      | 50.0      | 4.4  |
| FedLP              | 0.0    | **60.0** | 0.0   | 0.0  | 0.0   | 6.7       | 50.0      | 6.2  |
| FedLAMA            | 0.0    | 0.0    | 0.0   | 0.0  | 0.0   | 0.0       | 0.0       | 7.3  |
| pFedLA             | 0.0    | 0.0    | 0.0   | 0.0  | 0.0   | 0.0       | 0.0       | 7.7  |
| PLayer-FL          | **100.0** | 0.0    | **100.0** | 0.0  | 25.0  | 20.0      | 50.0      | **3.4** |
| PLayer-FL-Random | 60.0   | 0.0    | 80.0  | 0.0  | 25.0  | 6.7       | 25.0      | 4.8  |

## Accuracy

Table: Accuracy and average rank. In **bold** is the top-performing model. Friedman rank test p-value $<5\times10^{-3}$

| Algorithm        | FMNIST     | EMNIST     | CIFAR      | ISIC       | Heart      | Sentiment  | mimic      | Avg Rank |
| :--------------- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- | :------- |
| Local            | 88.7 ± 0.9 | 84.2 ± 0.9 | 79.4 ± 1.3 | **68.9 ± 1.8** | 53.7 ± 1.0 | 79.8 ± 0.3 | 74.7 ± 2.1 | 6.7      |
| FedAvg           | 89.1 ± 0.5 | **86.5 ± 0.6** | 80.0 ± 0.6 | 66.4 ± 1.2 | 54.1 ± 0.7 | 78.8 ± 0.2 | 74.7 ± 0.6 | 6.4      |
| FedProx          | 89.2 ± 0.4 | 86.2 ± 0.9 | 80.2 ± 0.6 | 66.5 ± 1.6 | 52.9 ± 1.6 | 78.8 ± 0.2 | **75.7 ± 1.7** | 6.4      |
| pFedMe           | 88.2 ± 0.8 | 85.7 ± 0.9 | 67.4 ± 1.1 | 66.8 ± 0.6 | 55.0 ± 1.0 | 80.0 ± 0.2 | 73.3 ± 0.9 | 6.9      |
| Ditto            | 89.0 ± 0.6 | 85.6 ± 1.1 | 67.8 ± 0.8 | 66.0 ± 1.0 | **55.5 ± 1.0** | 80.3 ± 0.6 | 73.5 ± 0.9 | 7.0      |
| LocalAdaptation  | 89.3 ± 0.8 | 86.3 ± 0.6 | 80.0 ± 1.1 | 66.9 ± 0.2 | 53.8 ± 0.8 | 78.9 ± 0.3 | 74.0 ± 2.2 | 6.3      |
| FedBABU          | 89.2 ± 0.6 | 86.4 ± 0.7 | 80.7 ± 1.4 | 67.3 ± 1.6 | 54.2 ± 0.7 | **81.0 ± 0.2** | 74.9 ± 0.6 | 3.4      |
| FedLP            | 89.1 ± 0.6 | 86.3 ± 0.9 | 78.6 ± 1.6 | 66.6 ± 0.7 | 54.1 ± 0.7 | 79.4 ± 0.3 | 74.6 ± 1.3 | 7.0      |
| FedLama          | 86.3 ± 0.8 | 83.3 ± 0.6 | 63.4 ± 2.0 | 62.4 ± 1.5 | 54.7 ± 0.8 | 78.0 ± 0.1 | 75.2 ± 1.2 | 8.9      |
| pFedLA           | 70.8 ± 0.2 | 41.2 ± 2.7 | 37.1 ± 2.3 | 56.7 ± 1.6 | 52.6 ± 1.4 | 78.0 ± 0.0 | 74.7 ± 2.6 | 11.1     |
| PLayer-FL        | **89.8 ± 0.7** | 86.1 ± 0.8 | **81.5 ± 1.1** | 68.6 ± 0.8 | 55.4 ± 1.2 | 80.7 ± 0.7 | **75.7 ± 0.6** | **2.2** |
| PLayer-FL-Random | 89.2 ± 0.7 | 84.1 ± 0.8 | 81.2 ± 1.2 | 68.6 ± 0.8 | 54.4 ± 1.2 | 79.0 ± 0.4 | 74.5 ± 2.0 | 5.8      |

Table: Variance in clients' accuracy (fairness). In **bold** is the fairest model. Friedman rank test p-value $<5\times10^{-3}$

| Algorithm          | FMNIST          | EMNIST          | CIFAR           | ISIC            | Heart           | Sentiment       | mimic           | Avg Rank |
| :----------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :------- |
| FedProx            | $1.0\cdot 10^{-4}$ | $3.0\cdot 10^{-4}$ | $1.0\cdot 10^{-4}$ | $3.5\cdot 10^{-2}$ | $6.7\cdot 10^{-2}$ | $2.4\cdot 10^{-2}$ | $1.1\cdot 10^{-2}$ | 5.7      |
| pFedMe             | $\mathbf{1.0\cdot 10^{-5}}$ | $3.0\cdot 10^{-4}$ | $4.0\cdot 10^{-4}$ | $3.3\cdot 10^{-2}$ | $4.6\cdot 10^{-2}$ | $2.1\cdot 10^{-2}$ | $1.4\cdot 10^{-2}$ | 5.0      |
| ditto              | $1.0\cdot 10^{-4}$ | $2.0\cdot 10^{-4}$ | $3.0\cdot 10^{-4}$ | $3.5\cdot 10^{-2}$ | $5.7\cdot 10^{-2}$ | $2.3\cdot 10^{-2}$ | $1.3\cdot 10^{-2}$ | 5.3      |
| LocalAdaptation    | $1.0\cdot 10^{-4}$ | $2.0\cdot 10^{-4}$ | $2.0\cdot 10^{-4}$ | $3.2\cdot 10^{-2}$ | $5.7\cdot 10^{-2}$ | $2.4\cdot 10^{-2}$ | $1.4\cdot 10^{-2}$ | 5.5      |
| FedBABU            | $1.0\cdot 10^{-4}$ | $2.0\cdot 10^{-4}$ | $1.0\cdot 10^{-4}$ | $3.3\cdot 10^{-2}$ | $6.0\cdot 10^{-2}$ | $\mathbf{1.8\cdot 10^{-2}}$ | $1.2\cdot 10^{-2}$ | 4.9      |
| FedLP              | $1.0\cdot 10^{-4}$ | $2.0\cdot 10^{-4}$ | $1.0\cdot 10^{-4}$ | $3.6\cdot 10^{-2}$ | $5.7\cdot 10^{-2}$ | $2.2\cdot 10^{-2}$ | $1.5\cdot 10^{-2}$ | 5.3      |
| FedLama            | $1.0\cdot 10^{-4}$ | $2.0\cdot 10^{-4}$ | $3.0\cdot 10^{-4}$ | $4.2\cdot 10^{-3}$ | $6.3\cdot 10^{-2}$ | $2.6\cdot 10^{-2}$ | $1.3\cdot 10^{-2}$ | 7.1      |
| pFedLA             | $9.0\cdot 10^{-4}$ | $6.0\cdot 10^{-4}$ | $2.3\cdot 10^{-3}$ | $6.3\cdot 10^{-2}$ | $5.1\cdot 10^{-2}$ | $2.6\cdot 10^{-2}$ | $2.5\cdot 10^{-2}$ | 8.9      |
| PLayer-FL          | $1.0\cdot 10^{-4}$ | $\mathbf{1.0\cdot 10^{-4}}$ | $1.0\cdot 10^{-4}$ | $\mathbf{2.9\cdot 10^{-2}}$ | $4.7\cdot 10^{-2}$ | $1.9\cdot 10^{-2}$ | $1.3\cdot 10^{-2}$ | **2.5** |
| PLayer-FL-Random | $1.0\cdot 10^{-4}$ | $3.0\cdot 10^{-4}$ | $\mathbf{1.0\cdot 10^{-4}}$ | $\mathbf{2.9\cdot 10^{-2}}$ | $5.7\cdot 10^{-2}$ | $2.5\cdot 10^{-2}$ | $\mathbf{9.7\cdot 10^{-3}}$ | 4.9      |

Table: Incentivized Participation Rate using accuracy (%). In **bold** is model with highest IPR. Friedman rank test p-value $<5\times10^{-3}$

| Algorithm          | FMNIST | EMNIST | CIFAR | ISIC | Heart | Sent-140 | Mimic-III | Avg Rank |
| :----------------- | :----- | :----- | :---- | :--- | :---- | :------- | :-------- | :------- |
| FedProx            | 20.0   | **40.0** | 40.0  | 0.0  | 0.0   | 0.0      | 50.0      | 5.6      |
| pFedMe             | 40.0   | **40.0** | 0.0   | 0.0  | **25.0** | 6.7      | 0.0       | 5.6      |
| Ditto              | 60.0   | 0.0    | 0.0   | 0.0  | 0.0   | 6.7      | 0.0       | 6.8      |
| LocalAdaptation    | 60.0   | 40.0   | 60.0  | 0.0  | 0.0   | 0.0      | 25.0      | 5.3      |
| FedBABU            | 80.0   | 20.0   | 60.0  | 0.0  | 0.0   | **20.0** | **25.0** | 4.3      |
| FedLP              | 60.0   | 20.0   | 20.0  | 0.0  | 0.0   | 6.7      | **25.0** | 5.4      |
| FedLama            | 0.0    | 0.0    | 0.0   | 0.0  | 0.0   | 0.0      | 0.0       | 8.1      |
| pFedLA             | 0.0    | 0.0    | 0.0   | 0.0  | 25.0  | 0.0      | 50.0      | 6.5      |
| PLayer-FL          | **80.0** | 0.0    | **100.0** | **25.0** | **25.0** | **20.0** | **75.0** | **2.4** |
| PLayer-FL-Random | 40.0   | 0.0    | **100.0** | **25.0** | 0.0   | 6.7      | 25.0      | 4.9      |

## Loss

Table: Test loss ($\times 1\cdot10^{-4}$) and average rank. In **bold** is the top-performing model. Friedman rank test p-value $<5\times10^{-3}$

| Algorithm     | FMNIST     | EMNIST     | CIFAR      | ISIC       | Heart      | Sentiment  | mimic      | Avg Rank |
| :------------ | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- | :------- |
| Local client  | 34.7 ± 2.0 | 58.6 ± 3.7 | 60.7 ± 4.3 | **111.2 ± 0.3** | 29.6 ± 0.8 | 43.7 ± 0.3 | 71.4 ± 0.8 | 8.3      |
| FedAvg        | 30.7 ± 1.1 | 42.2 ± 2.8 | 57.4 ± 3.1 | 121.8 ± 1.6 | 27.5 ± 0.5 | 44.1 ± 0.2 | 65.8 ± 0.9 | 6.1      |
| FedProx       | 31.5 ± 1.9 | 41.6 ± 2.9 | 56.3 ± 1.3 | 121.6 ± 1.1 | 28.4 ± 0.9 | 43.9 ± 0.2 | 65.8 ± 1.1 | 6.3      |
| pFedMe        | 30.4 ± 2.3 | 42.9 ± 2.7 | 90.9 ± 2.8 | 118.0 ± 1.0 | 26.4 ± 0.9 | 42.2 ± 0.6 | 67.4 ± 0.8 | 5.1      |
| Ditto         | 30.8 ± 0.2 | 45.0 ± 3.0 | 89.9 ± 3.9 | 120.6 ± 2.5 | **26.2 ± 0.4** | 42.5 ± 0.6 | 67.3 ± 0.1 | 5.9      |
| LocalAdaptation | 30.5 ± 1.9 | **41.1 ± 3.4** | 57.0 ± 3.0 | 121.4 ± 1.3 | 28.1 ± 0.6 | 43.6 ± 0.1 | 65.7 ± 0.6 | 4.9      |
| FedBABU       | 30.0 ± 1.9 | 42.2 ± 3.5 | 55.2 ± 2.8 | 120.6 ± 1.5 | 29.1 ± 1.0 | 41.8 ± 0.2 | 65.5 ± 0.7 | 3.8      |
| FedLP         | 30.1 ± 1.0 | 43.1 ± 3.2 | 61.4 ± 3.6 | 128.7 ± 1.5 | 27.3 ± 0.5 | 42.9 ± 0.2 | 65.8 ± 0.6 | 6.7      |
| FedLAMA       | 37.6 ± 2.0 | 53.2 ± 2.3 | 102.4 ± 3.0 | 133.0 ± 1.8 | 27.2 ± 0.5 | 45.0 ± 0.3 | **65.2 ± 0.5** | 8.3      |
| pFedLA        | 88.6 ± 6.9 | 253.4 ± 11.2 | 176.2 ± 10.6 | 151.2 ± 1.8 | 36.1 ± 4.9 | 47.2 ± 0.6 | 75.4 ± 0.6 | 12.0     |
| PLayer-FL     | **27.7 ± 2.0** | 45.4 ± 3.9 | **52.6 ± 2.6** | 113.4 ± 0.6 | 26.7 ± 0.6 | **41.4 ± 0.4** | 66.2 ± 0.7 | **3.4** |
| PLayer-FL-Random | 33.3 ± 2.3 | 58.2 ± 1.8 | 55.6 ± 3.7 | 113.1 ± 1.6 | 27.9 ± 1.0 | 44.6 ± 0.7 | 69.9 ± 1.2 | 7.4      |

Table: Variance in clients' test loss (fairness). In **bold** is the fairest model. Friedman rank test p-value $<5\times10^{-3}$

| Algorithm          | FMNIST          | EMNIST          | CIFAR           | ISIC            | Heart           | Sent-140        | MIMIC-III       | Avg Rank |
| :----------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :-------------- | :------- |
| FedProx            | $1.1\cdot 10^{-3}$ | $1.7\cdot 10^{-3}$ | $1.1\cdot 10^{-3}$ | $3.8\cdot 10^{-1}$ | $1.8\cdot 10^{-2}$ | $5.5\cdot 10^{-2}$ | $4.7\cdot 10^{-2}$ | 6.4      |
| pFedMe             | $5.0\cdot 10^{-4}$ | $2.3\cdot 10^{-3}$ | $1.8\cdot 10^{-3}$ | $3.6\cdot 10^{-1}$ | $1.3\cdot 10^{-2}$ | $5.2\cdot 10^{-2}$ | $3.5\cdot 10^{-2}$ | 4.1      |
| Ditto              | $6.0\cdot 10^{-4}$ | $2.2\cdot 10^{-3}$ | $2.3\cdot 10^{-4}$ | $3.8\cdot 10^{-1}$ | $1.7\cdot 10^{-2}$ | $5.2\cdot 10^{-2}$ | $3.8\cdot 10^{-2}$ | 4.6      |
| LocalAdaptation    | $1.0\cdot 10^{-3}$ | $2.1\cdot 10^{-3}$ | $1.2\cdot 10^{-3}$ | $4.0\cdot 10^{-1}$ | $1.9\cdot 10^{-2}$ | $5.5\cdot 10^{-2}$ | $4.8\cdot 10^{-2}$ | 7.1      |
| FedBABU            | $4.0\cdot 10^{-4}$ | $\mathbf{1.2\cdot 10^{-3}}$ | $7.0\cdot 10^{-4}$ | $4.1\cdot 10^{-1}$ | $2.1\cdot 10^{-2}$ | $\mathbf{4.6\cdot 10^{-2}}$ | $4.5\cdot 10^{-2}$ | 4.4      |
| FedLP              | $7.0\cdot 10^{-4}$ | $1.9\cdot 10^{-3}$ | $8.0\cdot 10^{-4}$ | $4.3\cdot 10^{-1}$ | $1.7\cdot 10^{-2}$ | $5.2\cdot 10^{-2}$ | $4.3\cdot 10^{-2}$ | 5.1      |
| FedLAMA            | $\mathbf{2.0\cdot 10^{-4}}$ | $2.9\cdot 10^{-3}$ | $2.4\cdot 10^{-3}$ | $4.6\cdot 10^{-1}$ | $1.8\cdot 10^{-2}$ | $5.3\cdot 10^{-2}$ | $4.3\cdot 10^{-2}$ | 6.7      |
| pFedLA             | $1.0\cdot 10^{-3}$ | $2.3\cdot 10^{-2}$ | $5.1\cdot 10^{-2}$ | $5.7\cdot 10^{-1}$ | $2.6\cdot 10^{-1}$ | $5.1\cdot 10^{-2}$ | $\mathbf{3.2\cdot 10^{-2}}$ | 7.6      |
| PLayer-FL          | $7.0\cdot 10^{-4}$ | $2.5\cdot 10^{-3}$ | $4.0\cdot 10^{-4}$ | $\mathbf{3.2\cdot 10^{-1}}$ | $\mathbf{1.4\cdot 10^{-2}}$ | $5.1\cdot 10^{-2}$ | $4.2\cdot 10^{-2}$ | **3.6** |
| PLayer-FL-Random | $8.0\cdot 10^{-4}$ | $4.1\cdot 10^{-3}$ | $\mathbf{1.0\cdot 10^{-4}}$ | $\mathbf{3.2\cdot 10^{-1}}$ | $1.6\cdot 10^{-2}$ | $6.1\cdot 10^{-2}$ | $4.0\cdot 10^{-2}$ | 5.2      |

Table: Incentivized Participation Rate using test loss (%) and Average Algorithm Rank. Friedman rank test p-value $=0.084$

| Algorithm          | FMNIST | EMNIST | CIFAR | ISIC | Heart | Sent-140 | Mimic-III | Avg Rank |
| :----------------- | :----- | :----- | :---- | :--- | :---- | :------- | :-------- | :------- |
| FedProx            | 40.0   | **80.0** | 40.0  | 0.0  | 25.0  | 33.3     | 25.0      | 5.1      |
| pFedMe             | 40.0   | 40.0   | 0.0   | 0.0  | **100.0** | 33.3     | 0.0       | 5.7      |
| Ditto              | 40.0   | 0.0    | 0.0   | 0.0  | 75.0  | 33.3     | 0.0       | 6.5      |
| LocalAdaptation    | 20.0   | 60.0   | 80.0  | 0.0  | 25.0  | 40.0     | 25.0      | 4.8      |
| FedBABU            | 80.0   | 40.0   | 60.0  | 0.0  | 0.0   | 40.0     | 25.0      | 4.6      |
| FedLP              | 60.0   | **80.0** | 20.0  | 0.0  | 75.0  | 40.0     | 50.0      | 3.6      |
| FedLAMA            | 0.0    | 0.0    | 0.0   | 0.0  | 50.0  | 20.0     | **75.0** | 6.7      |
| pFedLA             | 0.0    | 0.0    | 0.0   | 0.0  | 0.0   | 6.7      | 0.0       | 8.5      |
| PLayer-FL          | **80.0** | 0.0    | **100.0** | **50.0** | **100.0** | 40.0     | 0.0       | **3.5** |
| PLayer-FL-Random | 0.0    | 0.0    | **100.0** | **50.0** | 50.0  | 20.0     | 0.0       | 6.0      |