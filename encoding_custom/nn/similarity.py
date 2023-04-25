import torch.backends.cudnn
import torch.utils.data
from torchmetrics.functional import spearman_corrcoef

from utilities import generic_utils


class CKASimilarity:
    # Based on:
    # https://github.com/google-research/google-research/blob/34444253e9f57cd03364bc4e50057a5abe9bcf17/representation_similarity/Demo.ipynb

    def __init__(self, **kwargs):
        self.debiased = kwargs.get('debiased', True)
        self.feature_based = kwargs.get('feature_based', False)

    @staticmethod
    def gram_linear(x):
        """Compute Gram (kernel) matrix for a linear kernel.
        Args:
          x: A num_examples x num_features matrix of features.
        Returns:
          A num_examples x num_examples Gram matrix of examples.
        """
        return torch.matmul(x, x.T)

    @staticmethod
    def gram_rbf(x, threshold=1.0):
        """Compute Gram (kernel) matrix for an RBF kernel.
        Args:
          x: A num_examples x num_features matrix of features.
          threshold: Fraction of median Euclidean distance to use as RBF
           kernel bandwidth.
        Returns:
          A num_examples x num_examples Gram matrix of examples.
        """
        dot_products = torch.matmul(x, x.T)
        sq_norms = torch.diagonal(dot_products, 0)
        sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
        # Torch median gives slightly different result than numpy..
        sq_median_distance = torch.median(sq_distances)
        return torch.exp(-sq_distances / (
                2 * threshold ** 2 * sq_median_distance))

    @staticmethod
    def center_gram(gram, unbiased=False):
        """Center a symmetric Gram matrix.
        This is equvialent to centering the (possibly infinite-dimensional)
         features induced by the kernel before computing the Gram matrix.
        Args:
          gram: A num_examples x num_examples symmetric matrix.
          unbiased: Whether to adjust the Gram matrix in order to compute an
           unbiased estimate of HSIC. Note that this estimator may be negative.
        Returns:
          A symmetric matrix with centered columns and rows.
        """
        if not torch.allclose(gram, gram.T, equal_nan=True):
            raise ValueError('Input must be a symmetric matrix.')
        gram = gram.clone()

        if unbiased:
            n = gram.size(0)
            gram.fill_diagonal_(0)
            means = torch.sum(gram, 0, dtype=torch.float64) / (n - 2)
            means -= torch.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            gram.fill_diagonal_(0)
        else:
            means = torch.mean(gram, 0, dtype=torch.float64)
            means -= torch.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram

    def cka(self, gram_x, gram_y):
        gram_x = self.center_gram(gram_x, unbiased=self.debiased)
        gram_y = self.center_gram(gram_y, unbiased=self.debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2
        # (biased variant) or n*(n-3) (unbiased variant), but this
        # cancels for CKA.
        scaled_hsic = gram_x.view(-1).dot(gram_y.view(-1))

        normalization_x = torch.norm(gram_x)
        normalization_y = torch.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    @staticmethod
    def debiased_dot_product_similarity(
            xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x,
            squared_norm_y, n):
        return (xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
                + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

    def feature_space_linear_cka(self, features_x, features_y):
        """Compute CKA with a linear kernel, in feature space.
        This is typically faster than computing the Gram matrix when
         there are fewer features than examples.
        Args:
          features_x: A num_examples x num_features matrix of features.
          features_y: A num_examples x num_features matrix of features.
        Returns:
          The value of CKA between X and Y.
        """
        features_x -= torch.mean(features_x, dim=0, keepdim=True)
        features_y -= torch.mean(features_y, dim=0, keepdim=True)

        dot_product_similarity = torch.norm(
            torch.matmul(features_x.T, features_y)) ** 2

        normalization_x = torch.norm(torch.matmul(features_x.T, features_x))
        normalization_y = torch.norm(torch.matmul(features_y.T, features_y))

        if self.debiased:
            n = features_x.size(0)
            # Equivalent to np.sum(features_x ** 2, 1) but
            # avoids an intermediate array.
            sum_squared_rows_x = torch.einsum('ij,ij->i', features_x, features_x)
            sum_squared_rows_y = torch.einsum('ij,ij->i', features_y, features_y)
            squared_norm_x = torch.sum(sum_squared_rows_x)
            squared_norm_y = torch.sum(sum_squared_rows_y)

            dot_product_similarity = self.debiased_dot_product_similarity(
                dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
                squared_norm_x, squared_norm_y, n)
            normalization_x = torch.sqrt(self.debiased_dot_product_similarity(
                normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n))
            normalization_y = torch.sqrt(self.debiased_dot_product_similarity(
                normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n))

        return dot_product_similarity / (normalization_x * normalization_y)

    def calculate_similarity(self, feat_x, feat_y, kernel_type='linear',
                             replace_nan=True):
        if self.feature_based:
            assert kernel_type is 'linear'
            cka = self.feature_space_linear_cka(feat_x, feat_y)
        else:
            if kernel_type is 'linear':
                gram_x = self.gram_linear(feat_x)
                gram_y = self.gram_linear(feat_y)
                cka = self.cka(gram_x, gram_y)
            else:
                gram_x = self.gram_rbf(feat_x)
                gram_y = self.gram_rbf(feat_y)
                cka = self.cka(gram_x, gram_y)

        return torch.nan_to_num(cka) if replace_nan else cka
