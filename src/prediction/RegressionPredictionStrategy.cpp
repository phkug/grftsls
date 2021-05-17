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

#include <cmath>
#include <string>
#include "prediction/RegressionPredictionStrategy.h"

 // 1 denotes if constant is contained or not
const std::size_t RegressionPredictionStrategy::OUTCOME = 0;
const std::size_t RegressionPredictionStrategy::TREATMENT = 1;

// for prediction
const std::size_t RegressionPredictionStrategy::INSTRUMENT1 = 2;
const std::size_t RegressionPredictionStrategy::INSTRUMENT1_TREATMENT1 = 3;
const std::size_t RegressionPredictionStrategy::INSTRUMENT1_OUTCOME = 4;
const std::size_t RegressionPredictionStrategy::INSTRUMENT1_INSTRUMENT1 = 5;
const std::size_t RegressionPredictionStrategy::INSTRUMENT1_TREATMENT = 6;

// for variance (to maintain structure of moment condition)
const std::size_t RegressionPredictionStrategy::INSTRUMENT = 7;
const std::size_t RegressionPredictionStrategy::INSTRUMENT_TREATMENT = 8;
const std::size_t RegressionPredictionStrategy::INSTRUMENT_OUTCOME = 9;
const std::size_t RegressionPredictionStrategy::INSTRUMENT_INSTRUMENT = 10;
const std::size_t RegressionPredictionStrategy::INSTRUMENT_INSTRUMENT1 = 11;

const std::size_t NUM_TYPES = 12;

size_t RegressionPredictionStrategy::prediction_length() {
    return 1;
}

std::vector<double> RegressionPredictionStrategy::predict(const std::vector<Eigen::MatrixXd>& average) {
	Eigen::MatrixXd What_2 = average.at(INSTRUMENT1_TREATMENT1).transpose() * average.at(INSTRUMENT1_INSTRUMENT1).inverse() * average.at(INSTRUMENT1_TREATMENT1);
	Eigen::VectorXd What_Y = average.at(INSTRUMENT1_TREATMENT1).transpose() * average.at(INSTRUMENT1_INSTRUMENT1).inverse() * average.at(INSTRUMENT1_OUTCOME);

	Eigen::VectorXd beta = What_2.inverse() * What_Y;
	double tau = beta(1);

	return { tau };
}

std::vector<double> RegressionPredictionStrategy::compute_variance(
    const std::vector<Eigen::MatrixXd>& average,
    const PredictionValues& leaf_values,
    size_t ci_group_size) {

	int numiv = average.at(INSTRUMENT1_OUTCOME).size() - 1;

	// Get first stage effects
	Eigen::MatrixXd Z2 = average.at(INSTRUMENT1_INSTRUMENT1); // M+1 x M+1
	Eigen::VectorXd WZ = average.at(INSTRUMENT1_TREATMENT); // M+1 x 1
	Eigen::VectorXd gamma = Z2.inverse() * WZ; // (M+1)x1
	Eigen::VectorXd gamma1 = gamma.tail(numiv);
	double gamma0 = gamma(0);

	// Get effects of second stage (treatment and nuisance parameter)
	Eigen::MatrixXd What_2 = average.at(INSTRUMENT1_TREATMENT1).transpose() * average.at(INSTRUMENT1_INSTRUMENT1).inverse() * average.at(INSTRUMENT1_TREATMENT1); // 2x2
	Eigen::VectorXd What_Y = average.at(INSTRUMENT1_TREATMENT1).transpose() * average.at(INSTRUMENT1_INSTRUMENT1).inverse() * average.at(INSTRUMENT1_OUTCOME); // 2x1

	Eigen::VectorXd beta = What_2.inverse() * What_Y; // 2x1
	double mu = beta(0);
	double tau = beta(1);

	// Define relevant parameters
	double num_good_groups = 0; // for taking average 

	/* Get incredients to get H (p.18):
	1) Expected between group-variance: E[(1/l * sum_{b=1}^l (psi_b-psi))^2], where psi=0.
	   Therefore: get mean over all groups of squared mean of psi_b within bag
	2) Total variance of psi (variance across all trees) to finally get the within-group variance: E[1/B \sum_{b=1}^B (psi_b-psi)^2], where again psi=0.
	   Therefore: get mean of squared psi_b over all trees
	*/

	Eigen::MatrixXd psi_squared = Eigen::MatrixXd::Zero(numiv + 3, numiv + 3); // total variance: (M+3) x (M+3)
	Eigen::MatrixXd psi_grouped_squared = Eigen::MatrixXd::Zero(numiv + 3, numiv + 3); // between variance: (M+3 x M+3)

	// for each group
	for (size_t group = 0; group < leaf_values.get_num_nodes() / ci_group_size; ++group) { //leaf_values.get_num_nodes() = number of trees 
		//check if bag is good (values for each leave available)
		bool good_group = true;
		for (size_t j = 0; j < ci_group_size; ++j) {
			if (leaf_values.empty(group * ci_group_size + j)) {
				good_group = false;
			}
		}
		if (!good_group) continue; // if not, continute with next tree

		num_good_groups++; // if bag is well -> it is a "good group"

		Eigen::VectorXd group_psi = Eigen::VectorXd::Zero(numiv + 3); // (M+3)x1; to get [1/l * sum(Psi_b)] ^ 2 [between variance]

		// for each tree in (good) group
		for (size_t j = 0; j < ci_group_size; ++j) {

			size_t i = group * ci_group_size + j; // counter that accounts for nested-loop-structure
			const std::vector<Eigen::MatrixXd>& leaf_value = leaf_values.get_values(i); // get leaf values for j-th tree

			// generate psi
			Eigen::VectorXd psi(numiv + 3);

			double psi_1 = (gamma.transpose() * leaf_value.at(INSTRUMENT1_OUTCOME))(0)  // 1x(M+1) * (M+1)x1
				- (tau * gamma.transpose() * leaf_value.at(INSTRUMENT1_INSTRUMENT1) * gamma)(0) // 1x1 * 1x(M+1) * (M+1)x(M+1) * (M+1)x1
				- (mu * gamma.transpose() * leaf_value.at(INSTRUMENT1))(0);   // 1x1 * 1x(M+1) * (M+1)x1
			double psi_2 = (leaf_value.at(OUTCOME))(0)  // 1x1
				- (tau * gamma.transpose() * leaf_value.at(INSTRUMENT1))(0) - mu;  // 1x1 * 1x(M+1)* (M+1)x1 - 1x1
			Eigen::VectorXd psi_3 = leaf_value.at(INSTRUMENT_TREATMENT) - gamma0 * leaf_value.at(INSTRUMENT) - leaf_value.at(INSTRUMENT_INSTRUMENT) * gamma1;					// INCLUDED
			double psi_4 = leaf_value.at(TREATMENT)(0) - gamma0 - (leaf_value.at(INSTRUMENT).transpose() * gamma1)(0);										// INCLUDED

			//Eigen::VectorXd psi_43 = leaf_value.at(INSTRUMENT1_TREATMENT) - gamma.transpose() * leaf_value.at(INSTRUMENT1_INSTRUMENT1);
			// 1x(M+1) - 1x(M+1) * (M+1)x(M+1) 

			psi << psi_1, psi_2, psi_3, psi_4; // 1x1, 1x1, (M+1)x1

			// Between Variance: sum psi within group
			group_psi += psi;

			// Total Variance: sum squared psi over all trees
			psi_squared += psi * psi.transpose(); // psi_squared is defined outside of both loops [quantity that is needed at the end]
		}

		// Between Variance: take mean of psi
		group_psi /= ci_group_size;

		// Between Variance: square mean of psi and sum it up over all groups
		psi_grouped_squared += group_psi * group_psi.transpose(); // psi_grouped_squared is defined outside of both loops [quantity that is needed at the end]

	}

	// Between Variance: get average over all groups 
	psi_grouped_squared /= num_good_groups;

	// take average of squared sum of psi over all trees= number-of-groups * bag-size
	psi_squared /= (num_good_groups * ci_group_size);


	// generate V (p. 18)
	Eigen::MatrixXd E_ZZ = average.at(INSTRUMENT_INSTRUMENT); // get expectations conditional on X
	Eigen::MatrixXd E_Z1Z1 = average.at(INSTRUMENT1_INSTRUMENT1); // get expectations conditional on X
	Eigen::MatrixXd E_ZZ1 = average.at(INSTRUMENT_INSTRUMENT1); // get expectations conditional on X
	Eigen::VectorXd E_Z = average.at(INSTRUMENT);
	Eigen::VectorXd E_Z1 = average.at(INSTRUMENT1);
	Eigen::VectorXd E_ZY = average.at(INSTRUMENT_OUTCOME);
	Eigen::VectorXd E_Y = average.at(OUTCOME);

	Eigen::MatrixXd V(numiv + 3, numiv + 3);

	Eigen::VectorXd W_hat2 = gamma.transpose() * E_Z1Z1 * gamma;  // 1x(M+1) * (M+1)x(M+1) * (M+1)x1
	Eigen::VectorXd W_hat = gamma.transpose() * E_Z1; //1x(M+1) * (M+1)x1
	Eigen::VectorXd V_1_1toM = -E_ZY.transpose() + mu * E_Z.transpose() + 2 * tau* gamma.transpose() * E_ZZ1.transpose(); // 1xM + 1xM + 1xM+1 * M+1xM
	double V_1_M1 = -E_Y(0) + mu + 2 * tau * (gamma.transpose()*E_Z1)(0); // 1xM+1 * M+1x1
	Eigen::VectorXd V_2_1toM = tau * E_Z.transpose();  // M+1x1

	//Eigen::VectorXd dpsi_dgamma1 = -(E_Z1Y - tau * E_Z1Z1 * gamma - mu * E_Z1) + tau * E_Z1Z1 * gamma; // (M+1)x1- 1x1 *  (M+1)x(M+1) * (M+1)x1 - 1x1* (M+1)x1 + 1x1 * (M+1)x(M+1) * (M+1)x1
	//Eigen::VectorXd dpsi_dgamma2 = tau * E_Z1; // 1x1 * (M+1)*1

	V.col(0) << W_hat2, W_hat, Eigen::VectorXd::Zero(numiv + 1); // 1x1, 1x1, (M+1)x1
	V.col(1) << W_hat, 1, Eigen::VectorXd::Zero(numiv + 1); // 1x1, 1x1, (M+1)x1
	V.block(0, 2, 1, numiv + 1) << V_1_1toM.transpose(), V_1_M1;
	V.block(1, 2, 1, numiv + 1) << V_2_1toM.transpose(), tau;
	V.block(2, 2, numiv, numiv) << E_ZZ;
	V.block(2, numiv + 2, numiv, 1) << E_Z;
	V.block(numiv + 2, 2, 1, numiv + 1) << E_Z.transpose(), 1;

	//V.block(0,2, 1, numiv+1) << dpsi_dgamma1.transpose(); // 1x(M+1)
	//V.block(1,2,1, numiv+1) << dpsi_dgamma2.transpose(); // 1x(M+1)
	//V.block(2,2,numiv + 1, numiv + 1) << E_Z1Z1; // (M+1)x(M+1)

	Eigen::MatrixXd V_inv = V.inverse();

	// get between variance 
	Eigen::MatrixXd between_variance(numiv + 3, numiv + 3);
	between_variance = V_inv * psi_grouped_squared * V_inv.transpose();

	// get total variance
	Eigen::MatrixXd total_variance(numiv + 3, numiv + 3);
	total_variance = V_inv * psi_squared * V_inv.transpose();

	// get within variance (should be a scalar)
	Eigen::MatrixXd group_noise(numiv + 3, numiv + 3);
	group_noise = 1 / (ci_group_size - 1) * (total_variance - between_variance);

	// get variance of tau 
	double between_variance_tau = between_variance(0, 0);
	double group_noise_tau = group_noise(0, 0);

	double var_debiased = bayes_debiaser.debias(between_variance_tau, group_noise_tau, num_good_groups);

	return { var_debiased };
}


size_t RegressionPredictionStrategy::prediction_value_length() {
  return NUM_TYPES;
}

PredictionValues RegressionPredictionStrategy::precompute_prediction_values(
    const std::vector<std::vector<size_t>>& leaf_samples,
    const Data* data) {
	size_t num_leaves = leaf_samples.size();
	std::vector<std::vector<Eigen::MatrixXd>> values(num_leaves); // leaf[type[values]]] 

	for (size_t i = 0; i < leaf_samples.size(); ++i) { // for each leaf
		size_t leaf_size = leaf_samples[i].size();
		if (leaf_size == 0) {
			continue;
		}

		std::vector<Eigen::MatrixXd>& value = values[i]; // type & is a reference (is a memory location, can basically understood as an alias of a variable)
		value.resize(NUM_TYPES);

		double sum_Y = 0; // 1x1
		double sum_W = 0;  // 1x1
		Eigen::VectorXd sum_Z1 = Eigen::VectorXd::Zero(data->get_number_instruments() + 1);	// (M+1)x1
		Eigen::VectorXd sum_Z = Eigen::VectorXd::Zero(data->get_number_instruments());	// Mx1
		Eigen::MatrixXd sum_Z1W1 = Eigen::MatrixXd::Zero(data->get_number_instruments() + 1, 2); // (M+1)x1 * 1x2 = (M+1)x2
		Eigen::VectorXd sum_Z1W = Eigen::VectorXd::Zero(data->get_number_instruments() + 1);    // (M+1)x1
		Eigen::MatrixXd sum_ZW = Eigen::VectorXd::Zero(data->get_number_instruments()); // (M+1)x1 * 1x2 = (M+1)x2
		Eigen::VectorXd sum_Z1Y = Eigen::VectorXd::Zero(data->get_number_instruments() + 1);	// (M+1)x1
		Eigen::VectorXd sum_ZY = Eigen::VectorXd::Zero(data->get_number_instruments());	// M+x1
		Eigen::MatrixXd sum_Z1Z1 = Eigen::MatrixXd::Zero(data->get_number_instruments() + 1, data->get_number_instruments() + 1); // (M+1)x(M+1)
		Eigen::MatrixXd sum_ZZ = Eigen::MatrixXd::Zero(data->get_number_instruments(), data->get_number_instruments());  // MxM
		Eigen::MatrixXd sum_Z1Z = Eigen::MatrixXd::Zero(data->get_number_instruments(), data->get_number_instruments() + 1); // MxM+1

		for (auto& sample : leaf_samples[i]) {	// for each sample in that leaf

			Eigen::VectorXd Z1(data->get_number_instruments() + 1); // (M+1)x1
			Eigen::VectorXd Z = data->get_instrument(sample);
			Z1 << 1, Z;

			Eigen::VectorXd W1(2); // 2x1
			double W = data->get_treatment(sample); // 1x1
			W1 << 1, W;

			sum_Y += data->get_outcome(sample);   // 1x1
			sum_W += data->get_treatment(sample); // 1x1
			sum_Z1 += Z1; // (M+1)x1
			sum_Z += Z; // Mx1																			
			sum_Z1W1 += Z1 * W1.transpose(); // (M+1)x2
			sum_Z1W += Z1 * W; // (M+1)x1
			sum_ZW += Z * W; //  Mx1																	
			sum_Z1Y += Z1 * data->get_outcome(sample); // (M+1)x1
			sum_ZY += Z * data->get_outcome(sample); //  Mx1						
			sum_Z1Z1 += Z1 * Z1.transpose(); // (M+1)x(M+1)
			sum_ZZ += Z * Z.transpose();  //  MxM													    
			sum_Z1Z += Z * Z1.transpose();   //  MxM+1													    

		}

		Eigen::VectorXd eigen_sumY(1);
		eigen_sumY << sum_Y;

		Eigen::VectorXd eigen_sumW(1);
		eigen_sumW << sum_W;

		value[OUTCOME] = eigen_sumY / leaf_size;
		value[TREATMENT] = eigen_sumW / leaf_size;
		value[INSTRUMENT1] = sum_Z1 / leaf_size;
		value[INSTRUMENT] = sum_Z / leaf_size;														
		value[INSTRUMENT1_TREATMENT1] = sum_Z1W1 / leaf_size;
		value[INSTRUMENT1_TREATMENT] = sum_Z1W / leaf_size;
		value[INSTRUMENT_TREATMENT] = sum_ZW / leaf_size;												
		value[INSTRUMENT1_OUTCOME] = sum_Z1Y / leaf_size;
		value[INSTRUMENT_OUTCOME] = sum_ZY / leaf_size;												
		value[INSTRUMENT1_INSTRUMENT1] = sum_Z1Z1 / leaf_size;
		value[INSTRUMENT_INSTRUMENT] = sum_ZZ / leaf_size;												
		value[INSTRUMENT_INSTRUMENT1] = sum_Z1Z / leaf_size;											
	}

	return PredictionValues(values, NUM_TYPES);
}

std::vector<std::pair<double, double>>  RegressionPredictionStrategy::compute_error(
    size_t sample,
    const std::vector<Eigen::MatrixXd>& average,
    const PredictionValues& leaf_values,
    const Data* data) {
  
	// get reduced form estimate Y = rho * Z....WAS MACHEN WIR MIT DER KONSTANTE?
	Eigen::VectorXd rho = average.at(INSTRUMENT_INSTRUMENT).inverse() * average.at(INSTRUMENT_OUTCOME);

	// get Outcome and Instrument values
	double outcome = data->get_outcome(sample);
	Eigen::VectorXd instrument = data->get_instrument(sample);

	// compute raw error
	Eigen::VectorXd rhoZ = rho.transpose() * (instrument - average.at(INSTRUMENT));
	double residual = outcome - average.at(OUTCOME)(0) - rhoZ(0);
	double error_raw = residual * residual;
	
	// Estimates the Monte Carlo bias of the raw error via the jackknife estimate of variance.
	size_t num_trees = 0; // get non-empty number of leaves (=used trees)
	for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
		if (leaf_values.empty(n)) {
			continue;
		}
		num_trees++;
	}

	// If the treatment effect estimate is due to less than 5 trees, do not attempt to estimate error,
    // as this quantity is unstable due to non-linearities.
	if (num_trees <= 5) {
		return { std::make_pair<double, double>(NAN, NAN) };
	}

	// Compute 'leave one tree out' treatment effect estimates, and use them get a jackknife estimate of the excess error.
	double error_bias = 0.0;
	for (size_t n = 0; n < leaf_values.get_num_nodes(); n++) {
		if (leaf_values.empty(n)) {
			continue;
		}
		
		const std::vector<Eigen::MatrixXd>& leaf_value = leaf_values.get_values(n);
		Eigen::MatrixXd outcome_loto                        = (num_trees *  average.at(OUTCOME)               - leaf_value.at(OUTCOME)) / (num_trees - 1);
		Eigen::MatrixXd instrument_loto            = (num_trees *  average.at(INSTRUMENT)            - leaf_value.at(INSTRUMENT)) / (num_trees - 1);
		Eigen::MatrixXd instrument_outcome_loto    = (num_trees *  average.at(INSTRUMENT_OUTCOME)    - leaf_value.at(INSTRUMENT_OUTCOME)) / (num_trees - 1);
		Eigen::MatrixXd instrument_instrument_loto = (num_trees *  average.at(INSTRUMENT_INSTRUMENT) - leaf_value.at(INSTRUMENT_INSTRUMENT)) / (num_trees - 1);
	
		Eigen::VectorXd rho  = instrument_instrument_loto.inverse() * instrument_outcome_loto;
		Eigen::VectorXd rhoZ = rho.transpose() * (instrument - average.at(INSTRUMENT));

	    double residual_loto = outcome - outcome_loto(0,0) - rhoZ(0);
		
     	error_bias += (residual_loto - residual) * (residual_loto - residual);
	}
 
	error_bias *= ((double)(num_trees - 1)) / num_trees;

	double debiased_error = error_raw - error_bias;

    auto output = std::pair<double, double>(error_raw, debiased_error);
    return { output };
}

