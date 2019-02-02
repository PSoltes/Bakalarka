#include <Rcpp.h>

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#define _USE_MATH_DEFINES
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <functional>
#include <cmath>
#include <random>


double levyFlightDist(double beta, double s)
{
	return ((beta - 1)*tgamma(beta - 1)*sin((M_PI*(beta - 1)) / 2)) / M_PI * pow(s, beta);
}

double randomFromInterval(double max, double min)
{
	return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

class Moth
{
public:
	Moth() = default;

	Moth(Rcpp::NumericVector newMothCoords)
		:_coords(newMothCoords)
	{
		_fitness = 0;
	}

	Rcpp::NumericVector getCoords()
	{
		return this->_coords;
	}

	Rcpp::NumericVector getFitness()
	{
		return this->_fitness;
	}

	void computeFitness(Rcpp::Function optFunction)
	{
		this->_fitness = Rcpp::as<double>(optFunction(this->_coords));
	}

	void levyFlightMove(double maxWalkStep, size_t currGen)
	{
		double levy = levyFlightDist(1.5, 2); // s = 2, zistit dobru hodnotu
		double alpha = maxWalkStep / pow(currGen, 2);

		for (auto coord : _coords)
		{
			coord = coord + levy * alpha;
		}


	}

	void EqFiveSixMove(double scaleFactor, double goldenRatio, const Moth& bestMoth)
	{
		for (size_t i = 0; i < _coords.size(); i++)
		{
			this->_coords[i] = scaleFactor * (this->_coords[i] + goldenRatio * (bestMoth._coords[i] - this->_coords[i]));
		}

	}

	const bool operator<(const Moth& rhs) const
	{
		return this->_fitness < rhs._fitness;
	}

	Moth bestMothOBL(Moth bestMoth, Rcpp::NumericVector upperBounds, Rcpp::NumericVector lowerBounds, Rcpp::Function optFunc)
	{
		Rcpp::NumericVector newMothCoords(0.0, this->_coords.size());

		for (size_t i = 0; i < this->_coords.size(); i++)
		{
			newMothCoords[i] = 2 * bestMoth._coords[i] - this->_coords[i];
			if (newMothCoords[i] < lowerBounds[i] || newMothCoords[i] > upperBounds[i])
			{
				if (upperBounds[i] < lowerBounds[i])
					throw Rcpp::exception("Incorrect bounds -> upper bound lower than lower bound");
				newMothCoords[i] = randomFromInterval(upperBounds[i], lowerBounds[i]);
			}
		}

		Moth returnMoth(newMothCoords);

		returnMoth.computeFitness(optFunc);

		return returnMoth;
	}

	Moth boundsOBL(Rcpp::NumericVector upperBounds, Rcpp::NumericVector lowerBounds, Rcpp::Function optFunc)
	{
		Rcpp::NumericVector newMothCoords(0.0, this->_coords.size());

		for (size_t i = 0; i < this->_coords.size(); i++)
		{
			newMothCoords[i] = upperBounds[i] + lowerBounds[i] - this->_coords[i];
		}

		Moth returnMoth(newMothCoords);

		returnMoth.computeFitness(optFunc);

		return returnMoth;

	}

private:
	Rcpp::NumericVector _coords;
	double _fitness;
};

Rcpp::List MothOpt(Rcpp::Function optFunc, int maxGeneration, double maxWalkStep, Rcpp::NumericVector upperBounds, Rcpp::NumericVector lowerBounds)
{
	size_t currentGeneration = 1;
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.5, 0.1);
	double goldenRatio = 1 + sqrt(5) / 2;
	double scaleFactor = distribution(generator);
	Moth bestMoth;
	std::vector<Moth> population; //initialize, not sure how

#pragma omp parallel for
	for (int i = 0; i < population.size(); i++)
	{
		population[i].computeFitness(optFunc);
	}

	std::sort(population.begin(), population.end());

	for (; currentGeneration < maxGeneration; currentGeneration++)
	{
		std::sort(population.begin(), population.end());
		int i = 0;
		bestMoth = population[0];
#pragma omp parallel for
		for (; i < population.size() / 2; i++)
		{
			population[i].levyFlightMove(maxWalkStep, currentGeneration);
			population[i].computeFitness(optFunc);
			try {
				Moth OBLMoth = population[i].bestMothOBL(bestMoth, upperBounds, lowerBounds, optFunc);

				if (OBLMoth < population[i])
				{
					std::swap(OBLMoth, population[i]);
				}
			}
			catch (const std::exception& e)
			{
				std::terminate(); //to do something useful
			}

		}
#pragma omp parallel for
		for (; i < population.size(); i++)
		{
			double randDecider = ((double)rand() / (RAND_MAX));

			if (randDecider < 0.5)
			{
				population[i].EqFiveSixMove(scaleFactor, goldenRatio, bestMoth);
			}
			else
			{
				population[i].EqFiveSixMove(scaleFactor, 1 / goldenRatio, bestMoth);
			}
			population[i].computeFitness(optFunc);
			Moth OBLMoth = population[i].boundsOBL(upperBounds, lowerBounds, optFunc);
			if (OBLMoth < population[i])
			{
				std::swap(OBLMoth, population[i]);
			}
		}

	}

	return Rcpp::List::create(
		_["Coords"] = bestMoth.getCoords(),
		_["Value"] = bestMoth.getFitness()
	);

}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
*/
