import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit


class MultiModalEvaluation(object):
    """
    Class designed to fit multiple guassians to a distribution.
    """

    def __init__(self, data):
        """        
        Parameters
        ----------
        data : array like
            The data we want to fit. Needs to be a list or array of 
            values we want to find the distribution of.
        
        Methods
        -------
        fit():
            Main method used to fit to different models defined in 
            get_model().
        get_model():
            Returns a dictionary of all the models available.
        get_params():
            Returns a list of list of parameters for the used model.
        Attributes
        ----------
        data : array like
            see parameters
        modality : integer
            Number of independent probablity distributions of the same
            kind fit to data.
        bins : integer
            bins used to histogram data
        density : boolean, default True
            Wether or not to use probablity density or absolute values.
            So far only pdf supported.
        params : array like
            List of fitted parameters
        cov : array 
            Covariance matrix as given by scipy.optimize.curve_fit()
        _stride : number of parameters in the model. 
        """
        self.data = data
        self.modality = None
        self.bins = None
        self.density = True
        self.params = []
        self.cov = None
        self._stride = None

    def fit(self, *args, model="standard", modality=1, bins=25, density=True, **kwargs):
        """
        Fit a one or more probability distributions to your data.
        *args takes the initial guesses for you model parameters for 
        each independent function, e.g. if you fit two gaussians with 
        two parameters each you will need 4 values.
        **kwargs ultimately gets passed to np.histogram() to specify 
        things apart from bins and density. 

        Parameters
        ----------
        model : str, optional
            See available models in get_model(), by default "standard" 
            (gaussian with mean and standard deviation)
        modality : int, optional
            Number of independent model functions to fit to data, 
            by default 1
        bins : int, optional
            How many bins to use for binning the data, by default 25
        density : bool, optional
            Wether or not to use probability density or absolute counts, 
            by default True
        """
        model_dict = self.get_model().get(model)
        self._stride = model_dict.get("parameters")
        self.model = model_dict.get("function")
        self.modality = modality
        self.bins = bins
        self.density = density
        x, y = self.bin_data(**kwargs)

        initial_guess = args
        self.params, self.cov = curve_fit(self.obj_fct, x, y, initial_guess)

    def obj_fct(self, x, *args):
        """
        Wrapper for objective function to accommodate multimodal fitting.
        When fitting more than one distribution, the individual pdf is
        divided by the number of distributions (=modality) when density
        is set to true. *args depends on model set.

        Parameters
        ----------
        x : float
            Target value.

        Returns
        -------
        float:
            objective function at value of interest x.
            
        """
        obj_fct = 0.0
        if self.density:
            factor = 1.0 / self.modality
        else:
            factor = 1.0
        for i in range(self.modality):
            this_args = args[i * self._stride : (i + 1) * self._stride]
            obj_fct += self.model(x, *this_args) * factor
        return obj_fct

    def bin_data(self, **kwargs):
        y, x = np.histogram(self.data, bins=self.bins, density=self.density, **kwargs)
        x = (x[1:] + x[:-1]) / 2.0
        return x, y

    def get_model(self):
        """
        standard: Usual two parameter guassian which is normalised to
                  one. 2 parameters.
        3-param-gauss: As above but with a factor A to differently 
                  weight the individual distributsons. 3 parameters

        Returns
        -------
        dictionary
            dictionary with function reference to model and number of 
            parameters
        """
        models = {
            "standard": {"function": norm.pdf, "parameters": 2},
            "3-param-gauss": {"function": self.tpmgauss, "parameters": 3},
        }
        return models

    def tpmgauss(self, x, A, mean, std):
        """
        Returns A * scipy.stats.norm.pdf(x, mean, std)
        """
        return A * norm.pdf(x, mean, std)

    def get_params(self):
        """
        Return a list of list of parameters
        """
        params = []
        for i in range(self.modality):
            params.append(self.params[i * self._stride : (i + 1) * self._stride])

        return params
