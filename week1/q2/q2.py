import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2
    centered_mean=init_array-init_array.mean(axis=0)

    # TODO: transform init_array to final_data using PCA
    centered_mean=init_array-init_array.mean(axis=0)
    covariance_matrix=np.cov(centered_mean,rowvar=False)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance_matrix)
    sorted_indices=np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues=eigenvalues[sorted_indices]
    sorted_eigenvectors=eigenvectors[:,sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :dimensions]
    
    #  Transform the data
    final_data = np.dot(centered_mean, selected_eigenvectors)
    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(final_data[:, 0], final_data[:, 1])
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig("out.png")
    plt.show()
    # END TODO
