import torch, gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood = gpytorch.likelihoods.GaussianLikelihood(), grid_size_ratio = None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if grid_size_ratio is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x, grid_size_ratio)
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                grid_size=grid_size, num_dims=1,
            )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def pred_mean(self, test_x):
        return self.likelihood(self.forward(torch.Tensor(test_x.astype(float)))).mean.detach().cpu().numpy()

    @staticmethod
    def optimized_gp(train_x, train_y, epochs = 50, learning_rate = 0.1, weight_decay=1e-6, grid_size_ratio = None):
        train_x = torch.Tensor(train_x.astype(float))
        train_y = torch.Tensor(train_y.astype(float))

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood, grid_size_ratio = grid_size_ratio)
        model.train_x, model.train_y = train_x, train_y
        model.fit(epochs = epochs, learning_rate = learning_rate, weight_decay=weight_decay)
        
        return model
    
    def fit(self, epochs = 1, learning_rate = 5e-4, weight_decay=1e-6):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=learning_rate, weight_decay = weight_decay)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)


        for i in range(epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                i + 1, epochs, loss.item(),
                #self.covar_module.base_kernel.lengthscale.item(),
                self.likelihood.noise.item()
            ))
            optimizer.step()
        self.eval()