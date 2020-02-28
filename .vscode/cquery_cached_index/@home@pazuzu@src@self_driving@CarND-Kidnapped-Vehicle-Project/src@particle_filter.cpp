/**
 * particle_filter.cpp
 *
 * Created on: Feb 26, 2020
 * Author: Gamayunov Aleksandr
 * 
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::default_random_engine;
using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  num_particles = 100;
  default_random_engine geng;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; i++)
  {
    Particle p;
    p.id = i;
    p.x = dist_x(geng);
    p.y = dist_y(geng);
    p.theta = dist_theta(geng);
    p.weight = 1.0;
    particles.push_back(p);
    weights.push_back(p.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  for (int i = 0; i < particles.size(); ++i)
  {
    std::normal_distribution<double> N_theta(0, std_pos[2]);
    std::normal_distribution<double> N_y(0, std_pos[1]);
    std::normal_distribution<double> N_x(0, std_pos[0]);

    if (fabs(yaw_rate) < 0.001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    std::default_random_engine geng;
    particles[i].x += N_x(geng);
    particles[i].y += N_y(geng);
    particles[i].theta += N_theta(geng);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  for (int i = 0; i < observations.size(); ++i)
  {
    double min_dist = std::numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); ++j)
    {
      double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      if (distance < min_dist)
      {
        min_dist = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  for (int i = 0; i < num_particles; ++i)
  {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    vector<LandmarkObs> predictions;
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
    {
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;

      if (dist(p_x, p_y, lm_x, lm_y) <= sensor_range)
      {
        predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }

    vector<LandmarkObs> trans_observ;
    for (int j = 0; j < observations.size(); ++j)
    {
      double trans_x = p_x + cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y;
      double trans_y = p_y + sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y;
      trans_observ.push_back(LandmarkObs{observations[j].id, trans_x, trans_y});
    }
    dataAssociation(predictions, trans_observ);
    particles[i].weight = 1.0;
    for (int j = 0; j < trans_observ.size(); ++j)
    {
      double obs_x = trans_observ[j].x;
      double obs_y = trans_observ[j].y;
      double pred_x;
      double pred_y;
      for (int n = 0; n < predictions.size(); n++)
      {
        if (predictions[n].id == trans_observ[j].id)
        {
          pred_x = predictions[n].x;
          pred_y = predictions[n].y;
        }
      }
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double exponent = exp(-(pow(pred_x - obs_x, 2) / (2 * pow(std_x, 2)) + (pow(pred_y - obs_y, 2) / (2 * pow(std_y, 2)))));
      double normalizer = 2 * M_PI * std_x * std_y;
      double new_weight = (1 / normalizer) * exponent;

      particles[i].weight *= new_weight;
    }
  }
}

void ParticleFilter::resample()
{
  std::default_random_engine gen;
  vector<Particle> new_particles;
  vector<double> weights;
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
  }
  std::uniform_int_distribution<int> uniintdist(0, num_particles - 1);
  auto index = uniintdist(gen);
  double max_weight = *max_element(weights.begin(), weights.end());
  std::uniform_real_distribution<double> unirealdist(0.0, max_weight);
  double beta = 0.0;
  for (int i = 0; i < num_particles; i++)
  {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);
  return s;
}