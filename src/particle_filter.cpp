/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles =20;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i = 0; i<num_particles; i++){
		Particle p;
		p.id = i;
		p.weight = 1.0;

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta =dist_theta(gen);

		particles.push_back(p);
		weights.push_back(p.weight);
		//cout <<"0 => init id-"<<p.id<<"  "<<"init px-"<< p.x<<"  "<<"init py-"<<p.y<<"  "<<"init ptheta-"<<p.theta<<endl;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x1(0, std_pos[0]);
	normal_distribution<double> dist_y1(0, std_pos[1]);
	normal_distribution<double> dist_theta1(0, std_pos[2]);

	for (int i = 0; i<num_particles;i++){
		if(fabs(yaw_rate)<0.0001){
			particles[i].x += velocity*cos(particles[i].theta)*delta_t;
			particles[i].y += velocity*sin(particles[i].theta)*delta_t;
		}
		else{
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate*delta_t;
		}
		particles[i].x += dist_x1(gen);
		particles[i].y += dist_y1(gen);
		particles[i].theta += dist_theta1(gen);
	
		//cout << "1 => Prediction id-" << particles[i].id << "	"<<"x-" << particles[i].x << "	" \

		 //           <<"y-" << particles[i].y << "  "<<"theta-" << particles[i].theta << " "<<"weight-" << particles[i].weight << endl;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i =0; i<observations.size();i++){
		LandmarkObs ob = observations[i];
		double min_dis = 10000;
		int id_near = -1;
		int pre_index;

		for(int j = 0; j<predicted.size();j++){
			LandmarkObs pre = predicted[j];
			double dis = (ob.x-pre.x)*(ob.x-pre.x)+(ob.y-pre.y)*(ob.y-pre.y);
			if(dis<min_dis){
				min_dis = dis;
				id_near = pre.id;
				pre_index = j;
			}
		}
		observations[i].id = id_near;
		//cout << "2 => dataAssociation "<<"  "<<i<<"-ob_x-"<<ob.x<<"  "<<"pre_x-"<<predicted[pre_index].x<<"  "\
		//<<"ob_y-"<<ob.y<<"  "<<"pre_y-"<<predicted[pre_index].y<<"  "<<"=> near pre id-" << id_near << endl;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double norm_sum = 0.0;
	for(int i=0; i<num_particles;i++){
		double px = particles[i].x;
		double py = particles[i].y;
		double ptheta = particles[i].theta;
		vector<LandmarkObs> predict;
		particles[i].weight = 1.0;
	
		for(int j=0; j<map_landmarks.landmark_list.size();j++){
			Map::single_landmark_s lm = map_landmarks.landmark_list[j];
			double land_x = lm.x_f;
			double land_y = lm.y_f;
			double land_id = lm.id_i;
			double distence = (px - land_x)*(px - land_x)+(py - land_y)*(py - land_y);
			if(distence <= sensor_range*sensor_range){
				LandmarkObs pre;
				pre.x = land_x;
				pre.y = land_y;
				pre.id = land_id;
				predict.push_back(pre);
				//cout<<"5 => prediction id-"<<pre.id<<"  "<<"pre_x-"<<pre.x<<"  "<<"pre_y"<<pre.y<<endl;
			}
		}
		vector<LandmarkObs> observations_trans;
		for(int k=0; k<observations.size(); k++){
			double ob_x = observations[k].x;
			double ob_y = observations[k].y;
			int ob_id = observations[k].id;
			double tr_y = py +  sin(ptheta)*ob_x + cos(ptheta)*ob_y;
			double tr_x = px + cos(ptheta)*ob_x - sin(ptheta)*ob_y;
			observations_trans.push_back(LandmarkObs{ob_id, tr_x, tr_y});
		}

		dataAssociation(predict, observations_trans);

		//double prob = 1.0;
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		double nomer = 1.0/(2*M_PI*std_x*std_y);
		for(int l=0;l< observations_trans.size();l++){
			double ob_tr_x = observations_trans[l].x;
			double ob_tr_y = observations_trans[l].y;
			int ob_tr_id = observations_trans[l].id;

			for(int m=0; m<predict.size(); m++){
				double pred_x = predict[m].x;
			        double pred_y = predict[m].y;
				double pred_id = predict[m].id;
				if(ob_tr_id == pred_id){
					double difer_x = (ob_tr_x - pred_x)*(ob_tr_x - pred_x);
					double difer_y = (ob_tr_y - pred_y)*(ob_tr_y - pred_y);
					double wei = nomer*exp(-1.0*(difer_x/(2*std_x*std_x) + difer_y/(2*std_y*std_y)));
					 particles[i].weight *= wei;

					/*if(wei !=0){
						particles[i].weight *= wei;
					}
					else{particles[i].weight *= 0.1E-10;}*/
				}
			}
		}
		norm_sum += particles[i].weight;
		//cout<<"3 => weight id-"<<particles[i].id<<"  "<<"weight-"<<particles[i].weight<<endl;
	}
	for(int i=0; i<num_particles;i++){
		particles[i].weight /=norm_sum;
		weights[i] =  particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	uniform_int_distribution<int> un_int_dist(0, num_particles-1);
	int index = un_int_dist(gen);
	double beta = 0.0;
	vector<Particle> n_particles;
	//vector<double> weights;
	/*for(int i =0; i<num_particles;i++){
		weights.push_back(particles[i].weight);
	}*/
	double wei_max = *max_element(weights.begin(), weights.end());
	for(int j=0; j<num_particles;j++){
		uniform_real_distribution<double> un_real_dist(0.0, wei_max);
		beta += 2.0* un_real_dist(gen);
		while(beta > weights[index]){
			beta -= weights[index];
			index = (index +1) %num_particles;
		}
		n_particles.push_back(particles[index]);
		//cout<<"4 => resample"<<"  "<<j<<"-resample index-"<<index<<"  "<<"weight-"<<particles[index].weight<<endl;
	}
	particles = n_particles;	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
