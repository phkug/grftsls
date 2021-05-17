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

#include "commons/utility.h"
#include "relabeling/NoopRelabelingStrategy.h"

std::unordered_map<size_t, double> NoopRelabelingStrategy::relabel(
    const std::vector<size_t>& samples,
    const Data* data) {
 
	size_t num_samples = samples.size();
	size_t num_instrument = data->get_number_instruments();

	// GET COEFFICIENTS IN PARENT NODE

	// FIRST STAGE [in parnt node -> no weights!]
	Eigen::VectorXd sum_ZW = Eigen::VectorXd::Zero(num_instrument + 1);
	Eigen::MatrixXd sum_ZZ = Eigen::MatrixXd::Zero(num_instrument + 1, num_instrument + 1);

	for (size_t sample : samples) {
		Eigen::VectorXd Z1(num_instrument + 1);
		Eigen::VectorXd Z =	data->get_instrument(sample);
		Z1 << 1, Z;

		sum_ZW += Z1 * data->get_treatment(sample);
		sum_ZZ += Z1 * Z1.transpose();
	}

	if (equal_doubles(sum_ZZ.determinant(), 0.0, 1.0e-10)) {
		return std::unordered_map<size_t, double>(); // Signals that we should not perform a split.
	}

	Eigen::VectorXd gamma = sum_ZZ.inverse()*sum_ZW;	// gamma0 and gamma1

	std::unordered_map<size_t, double> W_hat;			// get first-stage predictions
	for (size_t sample : samples) {
		Eigen::VectorXd Z1(num_instrument + 1);
		Eigen::VectorXd Z = data->get_instrument(sample);
		Z1 << 1, Z;

		W_hat[sample] = Z1.transpose() * gamma;
	}

	// SECOND STAGE [NEU!]
	double total_outcome = 0;
	double total_what = 0;

	for (size_t sample : samples) {
		total_outcome += data->get_outcome(sample);
		total_what += W_hat[sample];
	}

	double average_outcome = total_outcome / num_samples;
	double average_what = total_what / num_samples;

	double numerator = 0;
	double denominator = 0;

	for (size_t sample : samples) {
		double outcome = data->get_outcome(sample);

		numerator += (outcome - average_outcome) * (W_hat[sample] - average_what);
		denominator += (W_hat[sample] - average_what) * (W_hat[sample] - average_what);
	}

	// get treatment effect
	double tau = numerator / denominator;

	// get intercept from second stage
	double mu = average_outcome - tau * average_what;

	// GENERATE Ap (mean in parent node of the gradient of moment condition evaluated in parent node)

	// get sum of gradient
	Eigen::MatrixXd sum_dpsi = Eigen::MatrixXd::Zero(num_instrument + 3, num_instrument + 3);

	for (size_t sample : samples) {

		double response = data->get_outcome(sample);			   // 1x1
		Eigen::VectorXd instrument = data->get_instrument(sample); // Mx1

		// FOR EACH SAMPLE COMPUTE Ap (gradient of moment condition)
		Eigen::MatrixXd dpsi(num_instrument + 3, num_instrument + 3);

		//get elements
		double W_hat2 = W_hat[sample] * W_hat[sample];

		Eigen::VectorXd dpsi_dgamma1_1 = -(response - tau * W_hat[sample] - mu) * instrument
			+ tau * W_hat[sample] * instrument;  // 1xM
		Eigen::VectorXd dpsi_dgamma1_2 = tau * instrument;																	  // 1xM
		Eigen::MatrixXd dpsi_dgamma1_3 = instrument * instrument.transpose();															  // MxM
		Eigen::VectorXd dpsi_dgamma1_4 = instrument;																		  // 1xM

		double dpsi_dgamma2_1 = -(response - tau * W_hat[sample] - mu) + tau * W_hat[sample];
		double dpsi_dgamma2_2 = tau;

		//fill columns of gradient matrix 
		dpsi.col(0) << W_hat2, W_hat[sample], Eigen::VectorXd::Zero(num_instrument + 1);	// first column
		dpsi.col(1) << W_hat[sample], 1, Eigen::VectorXd::Zero(num_instrument + 1);			// second column

		dpsi.block(0, 2, 1, num_instrument) = dpsi_dgamma1_1.transpose();								// third to M+2 column	(INFO: Block of size (p,q), starting at (i,j): matrix.block(i,j,p,q))				
		dpsi.block(1, 2, 1, num_instrument) = dpsi_dgamma1_2.transpose();
		dpsi.block(2, 2, num_instrument, num_instrument) = dpsi_dgamma1_3;
		dpsi.block(num_instrument + 2, 2, 1, num_instrument) = dpsi_dgamma1_4.transpose();

		dpsi.col(num_instrument + 2) << dpsi_dgamma2_1, dpsi_dgamma2_2, instrument, 1; // last column

		sum_dpsi += dpsi;	// sum iteratively
	}

	Eigen::MatrixXd Ap = sum_dpsi / num_samples;	// take mean

	if (equal_doubles(Ap.determinant(), 0.0, 1.0e-10)) {
		return std::unordered_map<size_t, double>(); // Signals that we should not perform a split.
	}

	// FIRST PART TO COMPUTE PSEUDO OUTCOME -> first row of inverse of Ap!
	Eigen::MatrixXd invAp(num_instrument + 3, num_instrument + 3);
	invAp = Ap.inverse();

	Eigen::MatrixXd xi_invAp(1, num_instrument + 3);
	xi_invAp.row(0) = invAp.row(0);

	// FOR EACH SAMPLE IN PARENT NODE CREATE PSY. THEN COMPUTE PSEUDO OUTCOME RHO AND STACK IT INTO A MAP
	std::unordered_map<size_t, double> relabeled_observations;

	for (size_t sample : samples) {

		double response = data->get_outcome(sample);
		double treatment = data->get_treatment(sample);
		Eigen::VectorXd instrument = data->get_instrument(sample); // Mx1

		// FOR EACH SAMPLE COMPUTE PSI (moment condition)
		Eigen::VectorXd psi(num_instrument + 3);  // initilaize psi ((M+3)x 1)

		double psi_1 = (response - tau * W_hat[sample] - mu) * W_hat[sample];	 // 1x1
		double psi_2 = response - tau * W_hat[sample] - mu;						 // 1x1
		Eigen::VectorXd psi_3 = (treatment - W_hat[sample]) * instrument;		 // Mx1
		double psi_4 = treatment - W_hat[sample];								 // 1x1

		psi << psi_1, psi_2, psi_3, psi_4;

		Eigen::VectorXd rho = -xi_invAp * psi;
		
		// COMPUTE PSEUDO OUTCOME
		relabeled_observations[sample] = rho(0);

	}

	return relabeled_observations;
}
