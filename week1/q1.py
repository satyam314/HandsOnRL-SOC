import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution
    rand_samples=np.random.uniform(0,1,num_samples)
    if distribution=="cauchy":
        peak_x=kwargs.get("peak_x",0)
        gamma=kwargs.get("gamma",1)
        samples.extend(peak_x+gamma*np.tan(np.pi*(rand_samples-0.5)))
    
    elif distribution == "exponential":
        l=kwargs.get("lambda",1)
        samples.extend(-np.log(1-rand_samples)/l)
    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        plt.hist(samples,bins=50, density=True, alpha=0.8)
        plt.title(f'Histogram for {distribution.capitalize()} distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(f"q1_{distribution}.png")
        # plt.clf
        # END TODO
