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

#include <Rcpp.h>
#include <sstream>
#include <vector>

#include "commons/globals.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainers.h"
#include "RcppUtilities.h"
#include "prediction/RegressionPredictionStrategy.h"
#include "relabeling/NoopRelabelingStrategy.h"


// TEST PREDICTION
// [[Rcpp::export]]
Rcpp::List test_tsls_prediction() {

	// define std::vector<Eigen::MatrixXd>& average
	std::vector<Eigen::MatrixXd> average_values;

	Eigen::VectorXd avgY(1);
	Eigen::VectorXd avgW(1);
	Eigen::VectorXd avgZ1(4);
	Eigen::MatrixXd avgZ1W1(4, 2);
	Eigen::VectorXd avgZ1Y(4);
	Eigen::MatrixXd avgZ1Z1(4, 4);
	Eigen::VectorXd avgZ1W(4);

	avgY << 2.23;
	avgW << 0.6;
	avgZ1 << 1, 3.4, 4.5, 0.5;
	avgZ1W1 << 0.3, 0.8, 0.1, 1.3, 2, 3.96, 0.89, 4.9;
	avgZ1Y << 1.2, 3.6, 4.6, 1.3;
	avgZ1Z1 << 1,	 2.56,	3.41, 9.95, 
			   2.56, 7.125, 5.5, 6.563,
			   3.41, 5.5,	1.23, 3.259,
			   9.95, 6.563, 3.259, 0.654;
	avgZ1W << 0.68, 5.453,  3.523, 7.586;

	average_values = { avgY, avgW, avgZ1, avgZ1W1, avgZ1Y, avgZ1Z1, avgZ1W };

	// call variance (without baysian debiasing)
	RegressionPredictionStrategy prediction_strategy;
	std::vector<double> compute_effect = prediction_strategy.predict(average_values);

	Prediction prediction(compute_effect);

	// Output
	std::vector<Prediction> output;
	output.push_back(prediction);
	Rcpp::List result = RcppUtilities::create_prediction_object(output);

	return result;
}

// TEST VARIANCE
// [[Rcpp::export]]
Rcpp::List test_tsls_compute_variance() {

	// define std::vector<Eigen::MatrixXd>& average
	std::vector<Eigen::MatrixXd> average_values;

	Eigen::VectorXd avgY(1);
	Eigen::VectorXd avgW(1);
	Eigen::VectorXd avgZ1(4);
	Eigen::MatrixXd avgZ1W1(4,2);
	Eigen::VectorXd avgZ1Y(4);
	Eigen::MatrixXd avgZ1Z1(4,4);
	Eigen::VectorXd avgZ1W(4);

	Eigen::VectorXd avgZ(3);
	Eigen::VectorXd avgZW(3);
	Eigen::VectorXd avgZY(3);
	Eigen::MatrixXd avgZZ(3, 3);
	Eigen::MatrixXd avgZZ1(3, 4);

	avgY << 2.23;
	avgW << 0.6;
	avgZ1 << 1, 3.4, 4.5, 0.5;
	avgZ1W1 << 0.3, 0.8, 0.1, 1.3, 2, 3.96, 0.89, 4.9;
	avgZ1Y << 1.2, 3.6, 4.6, 1.3;
	avgZ1Z1 << 1, 2.56, 3.41, 9.95,
		2.56, 7.125, 5.5, 6.563,
		3.41, 5.5, 1.23, 3.259,
		9.95, 6.563, 3.259, 0.654;
	avgZ1W << 0.68, 5.453, 3.523, 7.586;

	avgZ << 2.56, 2.389, 8.669;
	avgZW << 4.56, 3.698, 1.254;
	avgZY << 1.254, 3.687, 4.998;
	avgZZ << 1.3657, 2.3698, 1.2546,
			4.213, 6.124, 8.412,
			3.2115, 2.546, 6.214;

	avgZZ1 << 4.589, 3.256, 4.576, 4.266,
		7.4635, 6.5486, 4.235, 8.354,
		7.2235, 6.2553, 0.123, 5.689;

	average_values = { avgY, avgW, avgZ1, avgZ1W1, avgZ1Y, avgZ1Z1, avgZ1W, avgZ, avgZW, avgZY, avgZZ, avgZZ1 };


	// define PredictionValues& leaf_values

	//tree 1
	Eigen::VectorXd t1_Y(1);
	Eigen::VectorXd t1_W(1);
	Eigen::VectorXd t1_Z1(4);
	Eigen::MatrixXd t1_Z1W1(4, 2);
	Eigen::VectorXd t1_Z1Y(4);
	Eigen::MatrixXd t1_Z1Z1(4, 4);
	Eigen::VectorXd t1_Z1W(4);

	Eigen::VectorXd t1_Z(3);
	Eigen::VectorXd t1_ZW(3);
	Eigen::VectorXd t1_ZY(3);
	Eigen::MatrixXd t1_ZZ(3, 3);
	Eigen::MatrixXd t1_ZZ1(3, 4);

	t1_Y << 2.3;
	t1_W << 0.23;
	t1_Z1   << 1, 12.1, 5.7, 1.9;
	t1_Z1W1 << 0.5, 1.45, 3.8, 2.87, 6.98, 0.5, 1.9, 4.321;
	t1_Z1Y << 3.1, 5.8, 1.3, 5.789;
	t1_Z1Z1 << 1,	 2.35, 6.54, 7.89, 
			   2.35, 2.68,  7.86, 6.835,
			   6.54, 7.86,   2.4, 2.1,
			   7.89, 6.835,  2.1,  4.8;
	t1_Z1W << 0.9, 10.8, 2.6, 4.6;

	t1_Z << 0.596, 1.656, 0.369;
	t1_ZW << 7.56564, 4.366, 3.5684;
	t1_ZY << 4.213, 1.235, 7.696;
	t1_ZZ << 0.123, 2.0289, 1.0286,
		1.566, 0.286, 7.258,
		3.256, 1.268, 0.147;
	t1_ZZ1 << 4.7896, 4.4863, 1.2346, 0.124,
		2.145, 3.142, 2.145, 0.145,
		1.2354, 4.789, 0.1234, 3.2689;

	//tree 2
	Eigen::VectorXd t2_Y(1);
	Eigen::VectorXd t2_W(1);
	Eigen::VectorXd t2_Z1(4);
	Eigen::MatrixXd t2_Z1W1(4, 2);
	Eigen::VectorXd t2_Z1Y(4);
	Eigen::MatrixXd t2_Z1Z1(4, 4);
	Eigen::VectorXd t2_Z1W(4);

	Eigen::VectorXd t2_Z(3);
	Eigen::VectorXd t2_ZW(3);
	Eigen::VectorXd t2_ZY(3);
	Eigen::MatrixXd t2_ZZ(3, 3);
	Eigen::MatrixXd t2_ZZ1(3, 4);

	t2_Y << 9.7;
	t2_W << 0.87;
	t2_Z1 << 1, 0.6, 8.6, 3;
	t2_Z1W1 << 0.8, 1.69, 3.9, 4.7, 6.9, 10.1, 6.1, 1.7;
	t2_Z1Y << 6.9, 8.1, 2.3, 16.5;
	t2_Z1Z1 << 1,	6.8,  4.7,  3, 
			   6.8, 3.74, 5.78, 2.1,
			   4.7, 5.78, 6.97, 1.4,
				 3, 2.1, 1.4, 8.7;
	t2_Z1W << 4.58, 9.73, 2.14, 7.45;

	t2_Z << 0.145, 3.214, 7.245;
	t2_ZW << 1.234, 0.234, 7.256;
	t2_ZY << 0.245, 3.254, 1.1234;
	t2_ZZ << 6.145, 1.123, 3.5456,
		4.256, 5.567, 0.123,
		1.2634, 8.236, 9.2315;
	t2_ZZ1 << 8.756, 1.236, 1.458, 0.23,
		8.452, 1.234, 0.124, 4.123,
		4.12, 3.21, 0.145, 6.541;

	//tree 3
	Eigen::VectorXd t3_Y(1);
	Eigen::VectorXd t3_W(1);
	Eigen::VectorXd t3_Z1(4);
	Eigen::MatrixXd t3_Z1W1(4, 2);
	Eigen::VectorXd t3_Z1Y(4);
	Eigen::MatrixXd t3_Z1Z1(4, 4);
	Eigen::VectorXd t3_Z1W(4);

	Eigen::VectorXd t3_Z(3);
	Eigen::VectorXd t3_ZW(3);
	Eigen::VectorXd t3_ZY(3);
	Eigen::MatrixXd t3_ZZ(3, 3);
	Eigen::MatrixXd t3_ZZ1(3, 4);

	t3_Y << 4.78;
	t3_W << 0.69;
	t3_Z1 << 1, 3.7, 4.97, 9.8;
	t3_Z1W1 << 0.2, 8.7, 3.65, 1.8, 9.8, 7.6, 3.2, 1.2;
	t3_Z1Y << 4.94, 3.5, 1.97, 1.82;
	t3_Z1Z1 << 1, 2.3, 4.8, 7.9, 
			  2.3, 2.6, 4.9, 6.78,
			  4.8, 4.9, 4.6, 9.7,
			  7.9, 6.78, 9.7, 4.89;
	t3_Z1W << 0.8, 3.87, 2.78, 9.12;

	t3_Z << 5.36, 3.215, 0.126;
	t3_ZW << 7.896, 0.252, 7.895;
	t3_ZY << 2.145, 3.2145, 0.2145;
	t3_ZZ << 0.123, 0.257, 9.853,
		7.46, 2.365, 4.155,
		8.742, 5.231, 1.023;
	t3_ZZ1 << 1.234, 5.6521, 7.892, 4.213,
		0.214, 3.214, 2.012, 4.563,
		4.523, 2.126, 0.124, 3.214;

	std::vector<std::vector<Eigen::MatrixXd>> leaf_values;

	leaf_values = { {t1_Y, t1_W, t1_Z1, t1_Z1W1, t1_Z1Y, t1_Z1Z1, t1_Z1W, t1_Z, t1_ZW, t1_ZY, t1_ZZ, t1_ZZ1},
					{t2_Y, t2_W, t2_Z1, t2_Z1W1, t2_Z1Y, t2_Z1Z1, t2_Z1W, t2_Z, t2_ZW, t2_ZY, t2_ZZ, t2_ZZ1},
					{t3_Y, t3_W, t3_Z1, t3_Z1W1, t3_Z1Y, t3_Z1Z1, t3_Z1W, t3_Z, t3_ZW, t3_ZY, t3_ZZ, t3_ZZ1 }};

	PredictionValues prediction_values(leaf_values, 12); // values, number of trees, number of types

	// define bag size
	uint ci_group_size = 2;

	// call variance (without baysian debiasing)
	RegressionPredictionStrategy prediction_strategy;
	std::vector<double> variance = prediction_strategy.compute_variance(average_values, prediction_values, ci_group_size);

	Prediction prediction(variance);

	// Output
	std::vector<Prediction> output;
	output.push_back(prediction);
	Rcpp::List result = RcppUtilities::create_prediction_object(output);

	return result;
}

// TEST RELABELING
// [[Rcpp::export]]
Rcpp::List test_tsls_relabeling(std::vector<size_t> samples,
								Rcpp::NumericMatrix train_matrix,
								Eigen::SparseMatrix<double> sparse_train_matrix,
								size_t outcome_index,
								size_t treatment_index,
								Rcpp::IntegerVector input_instrument
								) {

	Data* data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
	std::vector<size_t> instrument_index = RcppUtilities::convert_integer_vector(input_instrument);

	data->set_outcome_index(outcome_index - 1);
	data->set_treatment_index(treatment_index - 1);
	data->set_instrument_index(instrument_index);

	NoopRelabelingStrategy relabeling_strategy;
	std::unordered_map<size_t, double> relabeld_outcomes_map = relabeling_strategy.relabel(samples, data);

	std::vector<double> relabeled_outcomes;
	for (size_t sample : samples) {
		relabeled_outcomes.push_back(relabeld_outcomes_map[sample]);
	}

	Prediction prediction(relabeled_outcomes);

	// Output
	std::vector<Prediction> relabed_outcomes_output;
	relabed_outcomes_output.push_back(prediction);
	Rcpp::List result = RcppUtilities::create_prediction_object(relabed_outcomes_output);

	return result;
}

// TEST DAD IMPROT
// [[Rcpp::export]]
double test_data(std::vector<size_t> samples,
							Rcpp::NumericMatrix train_matrix,
							Eigen::SparseMatrix<double> sparse_train_matrix,
							size_t outcome_index,
							size_t treatment_index,
						    Rcpp::IntegerVector input_instrument,
							int i) {

	Data* data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
	std::vector<size_t> instrument_index = RcppUtilities::convert_integer_vector(input_instrument);

	data->set_outcome_index(outcome_index - 1);
	data->set_treatment_index(treatment_index - 1);
	data->set_instrument_index(instrument_index);

	Eigen::VectorXd instrument = data->get_instrument(1); // Mx1
	double a = instrument(i);

	return a;
}
