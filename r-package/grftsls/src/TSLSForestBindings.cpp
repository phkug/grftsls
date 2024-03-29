#include <map>
#include <Rcpp.h>
#include <sstream>
#include <vector>

#include "commons/globals.h"
#include "Eigen/Sparse"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainers.h"
#include "RcppUtilities.h"

// [[Rcpp::export]]
Rcpp::List tsls_train(Rcpp::NumericMatrix train_matrix,
				      Eigen::SparseMatrix<double> sparse_train_matrix,
			          size_t outcome_index,
			          size_t treatment_index,
				      Rcpp::IntegerVector input_instrument,
			          size_t sample_weight_index,
					  bool use_sample_weights,
				      unsigned int mtry,
	  		          unsigned int num_trees,
			          unsigned int min_node_size,
				      double sample_fraction,
				      bool honesty,
				      double honesty_fraction,
				      size_t ci_group_size,
				      double alpha,
				      double imbalance_penalty,
				      std::vector<size_t> clusters,
		              unsigned int samples_per_cluster,
				      bool compute_oob_predictions,
				      unsigned int num_threads,
				      unsigned int seed) {

	ForestTrainer trainer = ForestTrainers::regression_trainer();

	Data* data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
	std::vector<size_t> instrument_index = RcppUtilities::convert_integer_vector(input_instrument);

	data->set_outcome_index(outcome_index - 1);
	data->set_treatment_index(treatment_index - 1);
	data->set_instrument_index(instrument_index);

	if (use_sample_weights) {
		data->set_weight_index(sample_weight_index - 1);
	}
	data->sort();

	ForestOptions options(num_trees, ci_group_size, sample_fraction, mtry, min_node_size, honesty,
		honesty_fraction, alpha, imbalance_penalty, num_threads, seed, clusters, samples_per_cluster);
	Forest forest = trainer.train(data, options);

	std::vector<Prediction> predictions;
	if (compute_oob_predictions) {
		ForestPredictor predictor = ForestPredictors::regression_predictor(num_threads);
		predictions = predictor.predict_oob(forest, data, false);
	}
	
	delete data;
	return RcppUtilities::create_forest_object(forest, predictions);
}


// [[Rcpp::export]]
Rcpp::List tsls_predict(Rcpp::List forest_object,
							 Rcpp::NumericMatrix train_matrix,
							 Eigen::SparseMatrix<double> sparse_train_matrix,
							 size_t outcome_index,
							 size_t treatment_index,
							 Rcpp::IntegerVector input_instrument,
							 Rcpp::NumericMatrix test_matrix,
						     Eigen::SparseMatrix<double> sparse_test_matrix,
							 unsigned int num_threads,
							 unsigned int estimate_variance) {

	Data* train_data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
	std::vector<size_t> instrument_index = RcppUtilities::convert_integer_vector(input_instrument);

	train_data->set_outcome_index(outcome_index - 1);
	train_data->set_treatment_index(treatment_index - 1);
	train_data->set_instrument_index(instrument_index);

	Data* data = RcppUtilities::convert_data(test_matrix, sparse_test_matrix);
	Forest forest = RcppUtilities::deserialize_forest(forest_object);

	ForestPredictor predictor = ForestPredictors::regression_predictor(num_threads);
	std::vector<Prediction> predictions = predictor.predict(forest, train_data, data, estimate_variance);
	Rcpp::List result = RcppUtilities::create_prediction_object(predictions);

	delete train_data;
	delete data;
	return result;
}

// [[Rcpp::export]]
Rcpp::List tsls_predict_oob(Rcpp::List forest_object,
								  Rcpp::NumericMatrix train_matrix,
							      Eigen::SparseMatrix<double> sparse_train_matrix,
								  size_t outcome_index,
								  size_t treatment_index,
							      Rcpp::IntegerVector input_instrument,
								  unsigned int num_threads,
								  bool estimate_variance) {
	Data* data = RcppUtilities::convert_data(train_matrix, sparse_train_matrix);
	std::vector<size_t> instrument_index = RcppUtilities::convert_integer_vector(input_instrument);

	data->set_outcome_index(outcome_index - 1);
	data->set_treatment_index(treatment_index - 1);
	data->set_instrument_index(instrument_index);

	Forest forest = RcppUtilities::deserialize_forest(forest_object);

	ForestPredictor predictor = ForestPredictors::regression_predictor(num_threads);
	std::vector<Prediction> predictions = predictor.predict_oob(forest, data, estimate_variance);
	Rcpp::List result = RcppUtilities::create_prediction_object(predictions);

	delete data;
	return result;
}

