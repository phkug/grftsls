/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#ifndef GRF_REGRESSIONPREDICTIONSTRATEGY_H
#define GRF_REGRESSIONPREDICTIONSTRATEGY_H

#include "commons/DefaultData.h"
#include "commons/Data.h"
#include "prediction/OptimizedPredictionStrategy.h"
#include "prediction/PredictionValues.h"
#include "ObjectiveBayesDebiaser.h"
#include <Eigen/Dense>

class RegressionPredictionStrategy: public OptimizedPredictionStrategy {
public:
  size_t prediction_value_length();

  PredictionValues precompute_prediction_values(const std::vector<std::vector<size_t>>& leaf_samples,
                                                const Data* data);

  size_t prediction_length();

  std::vector<double> predict(const std::vector<Eigen::MatrixXd>& average);

  std::vector<double> compute_variance(
      const std::vector<Eigen::MatrixXd>& average,
      const PredictionValues& leaf_values,
      size_t ci_group_size);

  std::vector<std::pair<double, double>> compute_error(
      size_t sample,
      const std::vector<Eigen::MatrixXd>& average,
      const PredictionValues& leaf_values,
      const Data* data);

private:
	static const std::size_t OUTCOME;
	static const std::size_t TREATMENT;
	static const std::size_t INSTRUMENT1;
	static const std::size_t INSTRUMENT1_TREATMENT1;
	static const std::size_t INSTRUMENT1_OUTCOME;
	static const std::size_t INSTRUMENT1_INSTRUMENT1;
	static const std::size_t INSTRUMENT1_TREATMENT;
	
	static const std::size_t INSTRUMENT;
	static const std::size_t INSTRUMENT_TREATMENT;
	static const std::size_t INSTRUMENT_OUTCOME;
	static const std::size_t INSTRUMENT_INSTRUMENT;
	static const std::size_t INSTRUMENT_INSTRUMENT1;
	ObjectiveBayesDebiaser bayes_debiaser;
};


#endif //GRF_REGRESSIONPREDICTIONSTRATEGY_H
