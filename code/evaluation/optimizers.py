"""
Custom optimizer implementations used in specific federated learning algorithms.

Includes:
- MySGD: A basic custom SGD implementation (potentially for demonstration or specific needs).
- pFedMeOptimizer: An optimizer specifically designed for the pFedMe algorithm,
  incorporating its update rule involving local weights and regularization.
"""
from configs import *


class MySGD(Optimizer):
    """
    A basic implementation of Stochastic Gradient Descent (SGD).

    This serves as a simplified example or might be used if specific
    modifications to standard SGD are needed. Supports a 'beta' parameter
    for alternative update rules, though standard SGD uses the learning rate.
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float):
        """
        Initializes the MySGD optimizer.

        Args:
            params (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize or
                                                    dicts defining parameter groups.
            lr (float): The learning rate.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    @torch.no_grad() # Disables gradient calculation for the update step itself
    def step(self, closure: Optional[callable] = None, beta: float = 0) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model
                                          and returns the loss. Optional for most optimizers.
            beta (float): An alternative scaling factor for the update. If non-zero,
                          it's used instead of the negative learning rate. Defaults to 0.

        Returns:
            Optional[float]: The loss computed by the closure, if provided. Otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): # Ensure gradients are enabled for closure
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue # Skip parameters without gradients

                d_p = p.grad # Get the gradient tensor

                # Apply the update rule
                if beta != 0:
                    # Use beta as the update step size (potentially for algorithms like Per-FedAvg)
                    # Update rule: p = p - beta * d_p
                    p.add_(d_p, alpha=-beta)
                else:
                    # Standard SGD update
                    # Update rule: p = p - lr * d_p
                    p.add_(d_p, alpha=-lr)

        return loss


class pFedMeOptimizer(Optimizer):
    """
    Custom optimizer implementing the update rule for the pFedMe algorithm.

    pFedMe involves updating personalized parameters (`theta`) using a rule that
    includes the standard gradient, a proximal term penalizing deviation from
    an approximated global model (`w`), and potentially L2 regularization.

    Note: This implementation assumes the `model_updated` parameters passed to `step`
    represent the reference model (`w` or `local_weight_updated` in the original code's comment)
    for the proximal term calculation.
    """
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 0.01, lamda: float = 0.1 , mu: float = 0.001):
        """
        Initializes the pFedMeOptimizer.

        Args:
            params (Iterable[torch.nn.Parameter]): Iterable of parameters of the personalized model (`theta`)
                                                   to optimize.
            lr (float): The learning rate (eta in pFedMe paper). Defaults to 0.01.
            lamda (float): The regularization parameter for the proximal term
                           (||theta - w||^2). Defaults to 0.1.
            mu (float): The L2 regularization parameter (||theta||^2). Defaults to 0.001.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lamda < 0.0:
            raise ValueError(f"Invalid lambda value: {lamda}")
        if mu < 0.0:
             raise ValueError(f"Invalid mu value: {mu}")
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, model_updated_params: Iterable[torch.Tensor], closure: Optional[callable] = None) -> Tuple[List[torch.nn.Parameter], Optional[float]]:
        """
        Performs a single pFedMe optimization step.

        Update rule for parameter `p` (representing `theta`):
        p = p - lr * (grad(p) + lamda * (p - w) + mu * p)
        where `w` is the corresponding parameter from `model_updated_params`.

        Args:
            model_updated_params (Iterable[torch.Tensor]): Parameters of the reference model (`w`),
                                                           used for the proximal term.
                                                           These should be detached tensors on the correct device.
            closure (callable, optional): A closure that re-evaluates the model
                                          and returns the loss.

        Returns:
            Tuple[List[torch.nn.Parameter], Optional[float]]: A tuple containing:
                - The list of updated parameters of the personalized model (`theta`).
                - The loss computed by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Convert iterable to list to ensure consistent iteration order if needed multiple times
        # Also ensures tensors are on the correct device
        ref_params_list = [w.detach().to(self.param_groups[0]['params'][0].device) for w in model_updated_params]

        updated_params_list = []
        for group in self.param_groups:
            lr = group['lr']
            lamda = group['lamda']
            mu = group['mu']

            group_params = [] # To return the params of this group
            # Use zip to iterate through personalized parameters and reference parameters
            for p, ref_weight in zip(group['params'], ref_params_list):
                if p.grad is None:
                    group_params.append(p)
                    continue # Skip if no gradient

                grad = p.grad # Gradient of the personalized parameter p (theta)

                # Apply the pFedMe update rule:
                # p.data = p.data - lr * (grad + lamda * (p.data - ref_weight.data) + mu * p.data)
                # Using add_ with alpha for potentially better efficiency/clarity:
                # Term1: -lr * grad
                p.add_(grad, alpha=-lr)
                # Term2: -lr * lamda * (p.data - ref_weight.data) -> -lr*lamda*p.data + lr*lamda*ref_weight.data
                p.add_(p.data, alpha=-lr * lamda)         # Add -lr*lamda*p.data
                p.add_(ref_weight.data, alpha=lr * lamda) # Add +lr*lamda*ref_weight.data
                # Term3: -lr * mu * p.data
                p.add_(p.data, alpha=-lr * mu)

                group_params.append(p) # Store reference to updated parameter

            updated_params_list.extend(group_params) # Collect all updated params

        # Let's return the list of all updated parameters managed by this optimizer.
        return updated_params_list, loss

    @torch.no_grad()
    def update_param(self, local_weight_updated: List[torch.Tensor], closure: Optional[callable] = None) -> List[torch.nn.Parameter]:
        """
        Directly sets the optimizer's parameters to match a given list of tensors.

        This method seems designed to overwrite the current personalized parameters (`theta`)
        with a new set of weights, potentially the `local_weight_updated` mentioned
        in the initial comment (representing `w` after its update step in pFedMe).
        Use with caution as it bypasses the gradient-based update.

        Args:
            local_weight_updated (List[torch.Tensor]): A list of tensors containing the new weights
                                                      to load into the optimizer's parameters.
                                                      Must match the order and shape of the
                                                      optimizer's managed parameters.
            closure (callable, optional): A closure that re-evaluates the model
                                          and returns the loss. (Not used in the update logic here).

        Returns:
            List[torch.nn.Parameter]: The list of parameters after their data has been updated.
        """
        loss = None
        if closure is not None:
            # Closure isn't used for the update, but might be called if provided,
            # though its result is ignored.
            with torch.enable_grad():
                loss = closure()

        all_updated_params = []
        # Ensure reference weights are on the correct device
        ref_weights = [w.detach().to(self.param_groups[0]['params'][0].device) for w in local_weight_updated]

        param_idx = 0
        for group in self.param_groups:
            group_params = []
            for p in group['params']:
                if param_idx < len(ref_weights):
                    # Directly copy the data from the provided tensor
                    p.data.copy_(ref_weights[param_idx].data)
                    group_params.append(p)
                    param_idx += 1
                else:
                    # This should not happen if local_weight_updated matches params
                    raise ValueError("Mismatch between number of optimizer parameters and provided weights.")
            all_updated_params.extend(group_params)

        return all_updated_params # Return the list of parameters with updated data