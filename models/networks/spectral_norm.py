import jittor as jt
import jittor.nn as nn
import jittor.init as init
class SpectralNorm2d(nn.Module):
    r"""2D Spectral Norm Module as described in `"Spectral Normalization
    for Generative Adversarial Networks by Miyato et. al." <https://arxiv.org/abs/1802.05957>`_
    The spectral norm is computed using ``power iterations``.

    Computation Steps:

    .. math:: v_{t + 1} = \frac{W^T W v_t}{||W^T W v_t||} = \frac{(W^T W)^t v}{||(W^T W)^t v||}
    .. math:: u_{t + 1} = W v_t
    .. math:: v_{t + 1} = W^T u_{t + 1}
    .. math:: Norm(W) = ||W v|| = u^T W v
    .. math:: Output = \frac{W}{Norm(W)} = \frac{W}{u^T W v}

    Args:
        module (nn.Module): The Module on which the Spectral Normalization needs to be
            applied.
        name (str, optional): The attribute of the ``module`` on which normalization needs to
            be performed.
        power_iterations (int, optional): Total number of iterations for the norm to converge.
            ``1`` is usually enough given the weights vary quite gradually.

    Example:
        .. code:: python

            >>> layer = SpectralNorm2d(Conv2d(3, 16, 1))
            >>> x = jt.rand(1, 3, 10, 10)
            >>> layer(x)
    """

    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm2d, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        # self.u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        self.u = nn.Parameter(init.gauss_(jt.zeros(height,dtype=jt.float32)),requires_grad=False)
        # self.u.requires_grad = False
        self.v = nn.Parameter(init.gauss_(jt.zeros(width,dtype=jt.float32)),requires_grad=False)
        # self.v.requires_grad=False
        # self.v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        self.u = self._l2normalize(self.u)
        self.v = self._l2normalize(self.v)
        self.w_bar = w.clone().astype(jt.float32)
        del self.module._parameters[self.name]

        self.eps = 1e-12


    def _l2normalize(self, x, eps=1e-12):
        r"""Function to calculate the ``L2 Normalized`` form of a Tensor

        Args:
            x (jt.Var): Tensor which needs to be normalized.
            eps (float, optional): A small value needed to avoid infinite values.

        Returns:
            Normalized form of the tensor ``x``.
        """
        return x / (jt.norm(x) + eps)

    # def execute(self, *args):
    #     r"""Computes the output of the ``module`` and appies spectral normalization to the
    #     ``name`` attribute of the ``module``.

    #     Returns:
    #         The output of the ``module``.
    #     """
    #     height = self.w_bar.size(0)
    #     # for _ in range(self.power_iterations):
    #     #     self.v = self._l2normalize(
    #     #         (jt.transpose(self.w_bar.view(height, -1)) @ self.u)
    #     #     )
    #     #     self.u = self._l2normalize(
    #     #         (self.w_bar.view(height, -1) @ self.v)
    #     #     )
    #     sigma = self.u @ (self.w_bar.view(height, -1) @ (self.v))
    #     setattr(self.module, self.name, (self.w_bar / sigma.expand_as(self.w_bar)))
    #     # setattr(self.module, self.name, self.w_bar)
    #     return self.module(*args)
    def execute(self, *args):
        r"""Computes the output of the ``module`` and appies spectral normalization to the
        ``name`` attribute of the ``module``.

        Returns:
            The output of the ``module``.
        """
        weight = self.compute_weight()
        setattr(self.module, self.name, weight)
        return self.module(*args)
    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = (jt.linalg.pinv(weight_mat.t() @ (weight_mat)) @ weight_mat.t() @  u.unsqueeze(1)).squeeze(1)
        # v = jt.linalg.multi_dot([weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]).squeeze(1)
        v = nn.matmul(v, target_sigma / (u @ (weight_mat @ v)))
        # return v.mul_(target_sigma / jt.dot(u, jt.mv(weight_mat, v)))
        return v
    def compute_weight(self,):
        weight = getattr(self.module, self.name)
        # weight = self.w_bar
        u = self.u
        v = self.v
        # u = getattr(module, self.name + '_u')
        # v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if self.module.is_training() or True:
            with jt.no_grad():
                for _ in range(self.power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = jt.normalize(jt.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                    u = jt.normalize(jt.matmul(weight_mat, v), dim=0, eps=self.eps)
                if self.power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone() 
        # sigma = jt.matmul(u, jt.matmul(weight_mat, v))
        # print("sigma", u.shape, v.shape, weight_mat.shape)
        sigma = u @ (weight_mat @ v)
        # print("sigma", u.shape, v.shape, weight_mat.shape)
        weight = weight / sigma
        return weight
    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

if __name__ == '__main__':
    a = jt.randn(1, 512)
    l = SpectralNorm2d(nn.Linear(512,3))
    print(l(a).shape)