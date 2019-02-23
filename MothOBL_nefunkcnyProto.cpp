
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
#include <omp.h>

typedef double(*vFunctionCall)(std::vector<double> x);


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

	Moth(std::vector<double> newMothCoords)
		:_coords(newMothCoords)
	{
		_fitness = 0;
	}

	std::vector<double> getCoords()
	{
		return this->_coords;
	}

	double getFitness()
	{
		return this->_fitness;
	}

	void computeFitness(vFunctionCall optFunction)
	{
		this->_fitness = optFunction(this->_coords);
	}

	void levyFlightMove(double maxWalkStep, size_t currGen)
	{
		double levy = levyFlightDist(1.5, 2.0); // s = 2, zistit dobru hodnotu
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

	Moth bestMothOBL(Moth bestMoth, std::vector<double> upperBounds, std::vector<double> lowerBounds, vFunctionCall optFunc)
	{
		std::vector<double> newMothCoords(this->_coords.size(), 0.0);

		for (size_t i = 0; i < this->_coords.size(); i++)
		{
			newMothCoords[i] = 2 * bestMoth._coords[i] - this->_coords[i];
			if (newMothCoords[i] < lowerBounds[i] || newMothCoords[i] > upperBounds[i])
			{
				if (upperBounds[i] < lowerBounds[i])
					throw std::exception("Incorrect bounds -> upper bound lower than lower bound");
				newMothCoords[i] = randomFromInterval(upperBounds[i], lowerBounds[i]);
			}
		}

		Moth returnMoth(newMothCoords);

		returnMoth.computeFitness(optFunc);

		return returnMoth;
	}

	Moth boundsOBL(std::vector<double> upperBounds, std::vector<double> lowerBounds, vFunctionCall optFunc)
	{
		std::vector<double> newMothCoords(this->_coords.size(), 0.0);

		for (size_t i = 0; i < this->_coords.size(); i++)
		{
			newMothCoords[i] = upperBounds[i] + lowerBounds[i] - this->_coords[i];
		}

		Moth returnMoth(newMothCoords);

		returnMoth.computeFitness(optFunc);

		return returnMoth;

	}

private:
	std::vector<double> _coords;
	double _fitness;
};

std::vector<Moth> generatePop(size_t popSize, const std::vector<double>& upperBounds, const std::vector<double>& lowerBounds)
{
	std::vector<Moth> returnVector;
	size_t dimensions = upperBounds.size();
	std::vector<std::vector<double>> popCoords(dimensions, std::vector<double>());
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	for (size_t i = 0; i < dimensions; i++)
	{
		double mean;
		//cause overflow
		if ((upperBounds[i] >= 0) ^ (lowerBounds[i] < 0)) //same sign?
		{
			mean = lowerBounds[i] + (upperBounds[i] - lowerBounds[i]) / 2;
		}
		else
		{
			mean = (upperBounds[i] + lowerBounds[i]) / 2;
		}

		double stddev = (upperBounds[i] - lowerBounds[i]) / 4;
		stddev = (stddev > DBL_MAX / 4 ? DBL_MAX / 4 : stddev);
		std::normal_distribution<double> distribution(mean, stddev);
		for (size_t j = 0; j < popSize; j++)
		{
			popCoords[i].push_back(distribution(gen));
		}
	}

	for (size_t i = 0; i < popSize; i++)
	{
		std::vector<double> currentMothCoords;
		for (size_t j = 0; j < dimensions; j++)
		{
			currentMothCoords.push_back(popCoords[j][i]);
		}

		returnVector.emplace_back(currentMothCoords);
	}

	return returnVector;
}
// [[Rcpp::export]]
Moth MothOpt(vFunctionCall optFunc, size_t popSize, int maxGeneration, double stopValue, double maxWalkStep, std::vector<double> upperBounds, std::vector<double> lowerBounds, int numThreads)
{
	omp_set_num_threads(numThreads);
	size_t currentGeneration = 1;
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<double> distribution(0.5, 0.1);
	double goldenRatio = 1 + sqrt(5) / 2;
	double scaleFactor = distribution(gen);
	Moth bestMoth;
	std::vector<Moth> population = generatePop(popSize, upperBounds, lowerBounds);

#pragma omp parallel for
	for (int i = 0; i < population.size(); i++)
	{
		population[i].computeFitness(optFunc);
	}

	std::sort(population.begin(), population.end());

	for (; currentGeneration < maxGeneration; currentGeneration++)
	{
		std::sort(population.begin(), population.end());
		bestMoth = population[0];
#pragma omp parallel for
		for (int i = 0; i < population.size() / 2; i++)
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
		for (int j = population.size() / 2 + 1; j < population.size(); j++)
		{
			double randDecider = ((double)rand() / (RAND_MAX));

			if (randDecider < 0.5)
			{
				population[j].EqFiveSixMove(scaleFactor, goldenRatio, bestMoth);
			}
			else
			{
				population[j].EqFiveSixMove(scaleFactor, 1 / goldenRatio, bestMoth);
			}
			population[j].computeFitness(optFunc);
			Moth OBLMoth = population[j].boundsOBL(upperBounds, lowerBounds, optFunc);
			if (OBLMoth < population[j])
			{
				std::swap(OBLMoth, population[j]);
			}
		}

		printf("%d:%lf\n", currentGeneration, bestMoth.getFitness());
	}

	return bestMoth;

}

double sphereFunction(std::vector<double> x)
{
	return std::transform_reduce(x.begin(),x.end(),x.begin, 0, std::multiplies<double>, std::plus<double>);
}

int main()
{
	Moth d = MothOpt(sphereFunction, 25, 50, 0.5, 0.0, std::vector<double>(2, 5.0), std::vector<double>(2, -5.0), 2);

	return 0;

}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
sphereFunction <- function(coords)
{
  result <- sum(coords^2)
  return (result)
}

result <- MothOpt(sphereFunction, 25, 100, 0.5, c(10.0,10.0), c(-10.0,-10.0))
result["Value"]
*/
