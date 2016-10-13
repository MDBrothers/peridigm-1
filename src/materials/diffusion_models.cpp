//! \file diffusion_models.cpp

//@HEADER
// ************************************************************************
//
//                             Peridigm
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?
// David J. Littlewood   djlittl@sandia.gov
// John A. Mitchell      jamitch@sandia.gov
// Michael L. Parks      mlparks@sandia.gov
// Stewart A. Silling    sasilli@sandia.gov
//
// ************************************************************************
//@HEADER

#include <complex>
#include <cmath>
#include <Sacado.hpp>
#include "diffusion_models.hpp"
#include "material_utilities.h"
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace MATERIAL_EVALUATION {

  template<typename ScalarT>
  void computePhaseOneDensityInPores
  (
    ScalarT* phaseOneDensityInPores,
    const ScalarT* porePressureYOverlap,
    const double* deltaTemperature,
    int numOwnedPoints
  ){
  // (pressure [Pa], Temperature [K], density: water density [kg/m3])
    ScalarT* densityOwned = phaseOneDensityInPores;
    const ScalarT* pressureOwned = porePressureYOverlap;
    const double* temperatureOwned = deltaTemperature;

    for(int p=0; p<numOwnedPoints; p++, densityOwned++, pressureOwned++, temperatureOwned++){
      const ScalarT pressureInMPa = (*pressureOwned)*1.0e-6;
      const ScalarT pressInMPaSquared = pressureInMPa*pressureInMPa;
      double tempSquared = (*temperatureOwned)*(*temperatureOwned);

      // Empirical relation supplied to the developer by Ouichi Hisanao
  	  *densityOwned = (-0.00000014569010515*pressInMPaSquared + 0.000046724532297*pressureInMPa - 0.0061488874609)*tempSquared
  		+ (0.000088493144499*pressInMPaSquared - 0.029002566308*pressureInMPa + 3.3982146161)*(*temperatureOwned)
  		- 0.013875092279*pressInMPaSquared + 4.9439957018*pressureInMPa + 530.4110022;
    }
  }

  template void computePhaseOneDensityInPores<double>
  (
    double* phaseOneDensityInPores,
    const double* porePressureYOverlap,
    const double* deltaTemperature,
    int numOwnedPoints
  );

  template void computePhaseOneDensityInPores<std::complex<double> >
  (
    std::complex<double>* phaseOneDensityInPores,
    const std::complex<double>* porePressureYOverlap,
    const double* deltaTemperature,
    int numOwnedPoints
  );

  template<typename ScalarT>
  void computePhaseOneDensityInFracture
  (
    ScalarT* phaseOneDensityInFracture,
    const ScalarT* fracturePRessureYOverlap,
    const double* deltaTemperature,
    int numOwnedPoints
  ){
  // (pressure [Pa], Temperature [K], density: water density [kg/m3])
    ScalarT* densityOwned = phaseOneDensityInFracture;
    const ScalarT* pressureOwned = fracturePRessureYOverlap;
    const double* temperatureOwned = deltaTemperature;

    for(int p=0; p<numOwnedPoints; p++, densityOwned++, pressureOwned++, temperatureOwned++){
      const ScalarT pressureInMPa = (*pressureOwned)*1.0e-6;
      const ScalarT pressInMPaSquared = pressureInMPa*pressureInMPa;
      double tempSquared = (*temperatureOwned)*(*temperatureOwned);

      // Empirical relation supplied to the developer by Ouichi Hisanao
  	  *densityOwned = (-0.00000014569010515*pressInMPaSquared + 0.000046724532297*pressureInMPa - 0.0061488874609)*tempSquared
  		+ (0.000088493144499*pressInMPaSquared - 0.029002566308*pressureInMPa + 3.3982146161)*(*temperatureOwned)
  		- 0.013875092279*pressInMPaSquared + 4.9439957018*pressureInMPa + 530.4110022;
    }
  }

  template void computePhaseOneDensityInFracture<double>
  (
    double* phaseOneDensityInFracture,
    const double* fracturePressureYOverlap,
    const double* deltaTemperature,
    int numOwnedPoints
  );

  template void computePhaseOneDensityInFracture<std::complex<double> >
  (
    std::complex<double> * phaseOneDensityInFracture,
    const std::complex<double> * fracturePRessureYOverlap,
    const double* deltaTemperature,
    int numOwnedPoints
  );

  namespace ONE_PHASE{

    template<typename ScalarT>
    void computeInternalFlow
    (
      const ScalarT* yOverlap,
      const ScalarT* porePressureYOverlap,
      const double* porePressureVOverlap,
      const ScalarT* fracturePressureYOverlap,
      const double* fracturePressureVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const ScalarT* matrixPorosityNP1,
      const double* matrixPorosityN,
      const ScalarT* fracturePorosityNP1,
      const double* fracturePorosityN,
    	const ScalarT* phaseOneDensityInPoresNP1,
    	const double* phaseOneDensityInPoresN,
    	const ScalarT* phaseOneDensityInFractureNP1,
    	const double* phaseOneDensityInFractureN,
      const ScalarT* breaklessDilatation,
      ScalarT* phaseOnePoreFlowOverlap,
      ScalarT* phaseOneFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature

    )
    {
      /*
       * Compute processor local contribution to internal fluid flow
       */
      const ScalarT *yOwned = yOverlap;
      const double *porePressureVOwned = porePressureVOverlap;
      const ScalarT *porePressureYOwned = porePressureYOverlap;
      const double *fracturePressureVOwned = fracturePressureVOverlap;
      const ScalarT *fracturePressureYOwned = fracturePressureYOverlap;

      const double *v = volumeOverlap;
      const double *deltaT = deltaTemperature;
      const double *damageOwned = damage;

      const ScalarT *matrixPorosityNP1Owned = matrixPorosityNP1;
      const double *matrixPorosityNOwned = matrixPorosityN;
      const ScalarT *fracturePorosityNP1Owned = fracturePorosityNP1;
      const double *fracturePorosityNOwned = fracturePorosityN;

    	const ScalarT* phaseOneDensityInPoresNP1Owned = phaseOneDensityInPoresNP1;
    	const double* phaseOneDensityInPoresNOwned = phaseOneDensityInPoresN;
    	const ScalarT* phaseOneDensityInFractureNP1Owned = phaseOneDensityInFractureNP1;
    	const double* phaseOneDensityInFractureNOwned = phaseOneDensityInFractureN;

      const ScalarT *thetaLocal = breaklessDilatation;
      ScalarT *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
      ScalarT *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;

      const int *neighPtr = localNeighborList;
      double cellVolume;
      ScalarT permeabilityTrace, phaseOnePorePerm, dPorePressure, dFracPressure;
      ScalarT dFracMinusPorePress, Y_dx, Y_dy, Y_dz, dY, fracWidth, fracPermeability;             // SA: fracWidth introduced
      ScalarT phaseOneFracPerm, scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow;
      ScalarT scalarPhaseOneFracToPoreFlow, omegaPores, omegaFrac;

      for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                        yOwned +=3, phaseOnePoreFlowOwned++,
                                        phaseOneFracFlowOwned++, deltaT++,
                                        damageOwned++, thetaLocal++,
                                        matrixPorosityNP1Owned++, fracturePorosityNP1Owned++,
                                        matrixPorosityNOwned++, fracturePorosityNOwned++,
    																		phaseOneDensityInFractureNP1Owned++, phaseOneDensityInPoresNP1Owned++,
    																		phaseOneDensityInFractureNOwned++, phaseOneDensityInPoresNOwned++,
                                        porePressureVOwned++, fracturePressureVOwned++){
        int numNeigh = *neighPtr; neighPtr++;
        double selfCellVolume = v[p];
        const ScalarT *Y = yOwned;
        const ScalarT *porePressureY = porePressureYOwned;
        const double *porePressureV = porePressureVOwned;
        const ScalarT *fracturePressureY = fracturePressureYOwned;
        const double *fracturePressureV = fracturePressureVOwned;

        // Fracture permeability
        fracWidth = 2.0*m_horizon/3.0*(*fracturePorosityNP1Owned);                   //TODO change this to grid spacing from m_horizon/3.0
        fracPermeability = fracWidth*fracWidth/12.0;

        dFracMinusPorePress = *fracturePressureY - *porePressureY;

        for(int n=0;n<numNeigh;n++,neighPtr++){
          int localId = *neighPtr;
          cellVolume = v[localId];
          const ScalarT *porePressureYP = &porePressureYOverlap[localId];
          const ScalarT *fracturePressureYP = &fracturePressureYOverlap[localId];
          const ScalarT *YP = &yOverlap[3*localId];
          const double *damageNeighbor = &damage[localId]; //TODO synchronize neighbor damage before force evaluation (this is aparently expensive though)

          Y_dx = *(YP+0) - *(Y+0);
          Y_dy = *(YP+1) - *(Y+1);
          Y_dz = *(YP+2) - *(Y+2);
          dY = sqrt(Y_dx*Y_dx+Y_dy*Y_dy+Y_dz*Y_dz);
          omegaPores = abs(1.0 - abs(dY/m_horizon));
          //Frac diffusion is a more local process than pore diffusion.
          omegaFrac = abs(1.0 - abs(dY/m_horizon_fracture));

          // Pressure potential
          dPorePressure = *porePressureYP - *porePressureY;
          dFracPressure = *fracturePressureYP - *fracturePressureY;

          // compute permeabilities
          phaseOneFracPerm = fracPermeability;
          /*
            Nonlocal permeability istropic tensor evaluation result
          */
          permeabilityTrace = (m_matrixPermeabilityXX + m_matrixPermeabilityYY + m_matrixPermeabilityZZ);
          phaseOnePorePerm = (m_matrixPermeabilityXX - 0.25 * permeabilityTrace) * Y_dx * Y_dx
                               + (m_matrixPermeabilityYY - 0.25 * permeabilityTrace) * Y_dy * Y_dy
                               + (m_matrixPermeabilityZZ - 0.25 * permeabilityTrace) * Y_dz * Y_dz;

          const double CORR_FACTOR_FRACTURE = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon_fracture*m_horizon_fracture*m_horizon_fracture);
          const double CORR_FACTOR_PORES = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon*m_horizon*m_horizon);

          // compute flow density
          // flow entering cell is positive
          scalarPhaseOnePoreFlow = omegaPores * CORR_FACTOR_PORES * (*phaseOneDensityInPoresNP1Owned) / m_phaseOneViscosity * phaseOnePorePerm / pow(dY, 4.0) * dPorePressure;
          scalarPhaseOneFracFlow = omegaFrac * CORR_FACTOR_FRACTURE * (*phaseOneDensityInFractureNP1Owned) / (2.0 * m_phaseOneViscosity) * phaseOneFracPerm / pow(dY, 2.0) * dFracPressure;

          // convert flow density to flow and account for reactions
          *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
          *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
          phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
          phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
        }
        double permeabilityAvg = (m_matrixPermeabilityXX + m_matrixPermeabilityYY + m_matrixPermeabilityZZ)/3.0;

        //Add in viscous and leakoff terms from mass conservation equation
        *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned -((*phaseOneDensityInPoresNP1Owned)*(*matrixPorosityNP1) - (*phaseOneDensityInPoresNOwned)*(*matrixPorosityN))/deltaTime + permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/2.0));
        *phaseOneFracFlowOwned = *phaseOneFracFlowOwned -((*phaseOneDensityInFractureNP1Owned)*(*fracturePorosityNP1) - (*phaseOneDensityInFractureNOwned)*(*fracturePorosityN))/deltaTime - permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/2.0));
      }
    }

    /** Explicit template instantiation for double. */
    template void computeInternalFlow<double>
    (
      const double* yOverlap,
      const double* porePressureYOverlap,
      const double* porePressureVOverlap,
      const double* fracturePressureYOverlap,
      const double* fracturePressureVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const double* matrixPorosityNP1,
      const double* matrixPorosityN,
      const double* fracturePorosityNP1,
      const double* fracturePorosityN,
    	const double* phaseOneDensityInPoresNP1,
    	const double* phaseOneDensityInPoresN,
    	const double* phaseOneDensityInFractureNP1,
    	const double* phaseOneDensityInFractureN,
    	const double* breaklessDilatation,
      double* phaseOnePoreFlowOverlap,
      double* phaseOneFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature
    );

    void computeInternalFlowComplex
    (
      const std::complex<double> * yOverlap,
      const std::complex<double> * porePressureYOverlap,
      const std::complex<double> * porePressureVOverlap,
      const std::complex<double> * fracturePressureYOverlap,
      const std::complex<double> * fracturePressureVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const std::complex<double> * matrixPorosityNP1,
      const double * matrixPorosityN,
      const std::complex<double> * fracturePorosityNP1,
      const double * fracturePorosityN,
    	const std::complex<double> * phaseOneDensityInPoresNP1,
    	const double * phaseOneDensityInPoresN,
    	const std::complex<double> * phaseOneDensityInFractureNP1,
    	const double * phaseOneDensityInFractureN,
      const std::complex<double>* breaklessDilatation,
      std::complex<double> * phaseOnePoreFlowOverlap,
      std::complex<double> * phaseOneFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature
    )
    {
      /*
       * Compute processor local contribution to internal fluid flow
       */
      const std::complex<double> *yOwned = yOverlap;
      const std::complex<double> *porePressureVOwned = porePressureVOverlap;
      const std::complex<double> *porePressureYOwned = porePressureYOverlap;
      const std::complex<double> *fracturePressureVOwned = fracturePressureVOverlap;
      const std::complex<double> *fracturePressureYOwned = fracturePressureYOverlap;

      const double *v = volumeOverlap;
      const double *deltaT = deltaTemperature;
      const double *damageOwned = damage;

      const std::complex<double> *matrixPorosityNP1Owned = matrixPorosityNP1;
      const double *matrixPorosityNOwned = matrixPorosityN;
      const std::complex<double> *fracturePorosityNP1Owned = fracturePorosityNP1;
      const double *fracturePorosityNOwned = fracturePorosityN;

      const std::complex<double>* phaseOneDensityInPoresNP1Owned = phaseOneDensityInPoresNP1;
      const double* phaseOneDensityInPoresNOwned = phaseOneDensityInPoresN;
      const std::complex<double>* phaseOneDensityInFractureNP1Owned = phaseOneDensityInFractureNP1;
      const double* phaseOneDensityInFractureNOwned = phaseOneDensityInFractureN;

      const std::complex<double> *thetaLocal = breaklessDilatation;
      std::complex<double> *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
      std::complex<double> *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;

      const int *neighPtr = localNeighborList;
      double cellVolume;
      std::complex<double> phaseOnePorePerm,  dPorePressure, dFracPressure;
      std::complex<double> dFracMinusPorePress, Y_dx, Y_dy, Y_dz, dY, fracPermeability;
      std::complex<double> phaseOneFracPerm, permeabilityTrace;
      std::complex<double> scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, phaseOneRelPermFrac;
      std::complex<double> scalarPhaseOneFracToPoreFlow, omegaPores, omegaFrac, fracWidth;

      for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                        yOwned +=3, phaseOnePoreFlowOwned++,
                                        phaseOneFracFlowOwned++, deltaT++,
                                        damageOwned++, thetaLocal++,
                                        matrixPorosityNP1Owned++, fracturePorosityNP1Owned++,
                                        matrixPorosityNOwned++, fracturePorosityNOwned++,
                                        phaseOneDensityInFractureNP1Owned++, phaseOneDensityInPoresNP1Owned++,
                                        phaseOneDensityInFractureNOwned++, phaseOneDensityInPoresNOwned++,
                                        porePressureVOwned++, fracturePressureVOwned++){
        int numNeigh = *neighPtr; neighPtr++;
        double selfCellVolume = v[p];
        const std::complex<double> *Y = yOwned;
        const std::complex<double> *porePressureY = porePressureYOwned;
        const std::complex<double> *porePressureV = porePressureVOwned;
        const std::complex<double> *fracturePressureY = fracturePressureYOwned;
        const std::complex<double> *fracturePressureV = fracturePressureVOwned;

        // Fracture permeability
        fracWidth = 2.0*m_horizon/3.0*(*fracturePorosityNP1Owned);  //TODO replace m_horizon/3.0 with actual grid spacing
        fracPermeability = fracWidth*fracWidth/12.0;

        dFracMinusPorePress = *fracturePressureY - *porePressureY;

        for(int n=0;n<numNeigh;n++,neighPtr++){
          int localId = *neighPtr;
          cellVolume = v[localId];
          const std::complex<double> *porePressureYP = &porePressureYOverlap[localId];
          const std::complex<double> *fracturePressureYP = &fracturePressureYOverlap[localId];
          const std::complex<double> *YP = &yOverlap[3*localId];
          const double *damageNeighbor = &damage[localId]; //TODO synchronize neighbor damage before force evaluation (this is aparently expensive though)

          Y_dx = *(YP+0) - *(Y+0);
          Y_dy = *(YP+1) - *(Y+1);
          Y_dz = *(YP+2) - *(Y+2);
          dY = sqrt(Y_dx*Y_dx+Y_dy*Y_dy+Y_dz*Y_dz);
          //NOTE I want to use std::complex<double>, which is why I circumvent the standard influence function code.
          //NOTE real part needs to be nonnegative.
          omegaPores =  std::complex<double>(1.0, 0.0) - std::abs(double(std::real(dY/m_horizon))) - std::complex<double>(0.0, std::imag(dY/m_horizon));
          //Frac diffusion is a more local process than pore diffusion.
          omegaFrac =  std::complex<double>(1.0, 0.0) - std::abs(double(std::real(dY/m_horizon_fracture))) - std::complex<double>(0.0, std::imag(dY/m_horizon_fracture));

          // Pressure potential
          dPorePressure = *porePressureYP - *porePressureY;
          dFracPressure = *fracturePressureYP - *fracturePressureY;

          // compute permeabilities
          phaseOneFracPerm = fracPermeability;
          /*
            Nonlocal permeability istropic tensor evaluation result
          */
          permeabilityTrace = (m_matrixPermeabilityXX + m_matrixPermeabilityYY + m_matrixPermeabilityZZ);
          phaseOnePorePerm = (m_matrixPermeabilityXX - 0.25 * permeabilityTrace) * Y_dx * Y_dx
                               + (m_matrixPermeabilityYY - 0.25 * permeabilityTrace) * Y_dy * Y_dy
                               + (m_matrixPermeabilityZZ - 0.25 * permeabilityTrace) * Y_dz * Y_dz;

          const double CORR_FACTOR_FRACTURE = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon_fracture*m_horizon_fracture*m_horizon_fracture);
          const double CORR_FACTOR_PORES = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon*m_horizon*m_horizon);

          // compute flow density
          // flow entering cell is positive
          scalarPhaseOnePoreFlow = omegaPores * CORR_FACTOR_PORES * (*phaseOneDensityInPoresNP1Owned) / m_phaseOneViscosity * phaseOnePorePerm / pow(dY, 4.0) * dPorePressure;
          scalarPhaseOneFracFlow = omegaFrac * CORR_FACTOR_FRACTURE * (*phaseOneDensityInFractureNP1Owned) / (2.0 * m_phaseOneViscosity) * phaseOneFracPerm / pow(dY, 2.0) * dFracPressure;

          // convert flow density to flow and account for reactions
          *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
          *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
          phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
          phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
        }
        double permeabilityAvg = (m_matrixPermeabilityXX + m_matrixPermeabilityYY + m_matrixPermeabilityZZ)/3.0;

        //Add in viscous and leakoff terms from mass conservation equation
        //NOTE it is assumed that grid spacing is 1/3 of the standard horizon in order to compute leakoff diffusion area
        *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned -((*phaseOneDensityInPoresNP1Owned)*(*matrixPorosityNP1) - (*phaseOneDensityInPoresNOwned)*(*matrixPorosityN))/deltaTime + permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/6.0));
        *phaseOneFracFlowOwned = *phaseOneFracFlowOwned -((*phaseOneDensityInFractureNP1Owned)*(*fracturePorosityNP1) - (*phaseOneDensityInFractureNOwned)*(*fracturePorosityN))/deltaTime - permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/6.0));
      }
    }

  }

  namespace TWO_PHASE{

    template<typename ScalarT>
    void computeInternalFlow
    (
      const ScalarT* yOverlap,
      const ScalarT* porePressureYOverlap,
      const double* porePressureVOverlap,
      const ScalarT* fracturePressureYOverlap,
      const double* fracturePressureVOverlap,
      const ScalarT* phaseOneSaturationPoresYOverlap,
      const double* phaseOneSaturationPoresVOverlap,
      const ScalarT* phaseOneSaturationFracYOverlap,
      const double* phaseOneSaturationFracVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const ScalarT* matrixPorosityNP1,
      const double* matrixPorosityN,
      const ScalarT* fracturePorosityNP1,
      const double* fracturePorosityN,
      const ScalarT* phaseOneDensityInPoresNP1,
      const double* phaseOneDensityInPoresN,
      const ScalarT* phaseOneDensityInFractureNP1,
      const double* phaseOneDensityInFractureN,
      const ScalarT* phaseTwoDensityInPoresNP1,
      const double* phaseTwoDensityInPoresN,
      const ScalarT* phaseTwoDensityInFractureNP1,
      const double* phaseTwoDensityInFractureN,
      const ScalarT* breaklessDilatationOwned,
      ScalarT* phaseOnePoreFlowOverlap,
      ScalarT* phaseOneFracFlowOverlap,
      ScalarT* phaseTwoPoreFlowOverlap,
      ScalarT* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature
    )
    {
      /*
       * Compute processor local contribution to internal fluid flow
       */

      const ScalarT *yOwned = yOverlap;
      const double *porePressureVOwned = porePressureVOverlap;
      const ScalarT *porePressureYOwned = porePressureYOverlap;
      const double *fracturePressureVOwned = fracturePressureVOverlap;
      const ScalarT *fracturePressureYOwned = fracturePressureYOverlap;
      const ScalarT *phaseOneSaturationPoresYOwned = phaseOneSaturationPoresYOverlap;
      const double *phaseOneSaturationPoresVOwned = phaseOneSaturationPoresVOverlap;
      const ScalarT *phaseOneSaturationFracYOwned = phaseOneSaturationFracYOverlap;
      const double *phaseOneSaturationFracVOwned = phaseOneSaturationFracVOverlap;

      const double *v = volumeOverlap;
      const double *deltaT = deltaTemperature;
      const double *damageOwned = damage;

      const ScalarT* matrixPorosityNP1Owned = matrixPorosityNP1;
      const double* matrixPorosityNOwned = matrixPorosityN;
      const ScalarT* fracturePorosityNP1Owned = fracturePorosityNP1;
      const double* fracturePorosityNOwned= fracturePorosityN;

      const ScalarT* phaseOneDensityInPoresNP1Owned = phaseOneDensityInPoresNP1;
      const double* phaseOneDensityInPoresNOwned = phaseOneDensityInPoresN;
      const ScalarT* phaseOneDensityInFractureNP1Owned = phaseOneDensityInFractureNP1;
      const double* phaseOneDensityInFractureNOwned = phaseOneDensityInFractureN;

      const ScalarT* phaseTwoDensityInPoresNP1Owned = phaseTwoDensityInPoresNP1;
      const double* phaseTwoDensityInPoresNOwned = phaseTwoDensityInPoresN;
      const ScalarT* phaseTwoDensityInFractureNP1Owned = phaseTwoDensityInFractureNP1;
      const double* phaseTwoDensityInFractureNOwned = phaseTwoDensityInFractureN;

      const ScalarT *thetaBreakless = breaklessDilatationOwned;

      ScalarT *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
      ScalarT *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;
      ScalarT *phaseTwoPoreFlowOwned = phaseTwoPoreFlowOverlap;
      ScalarT *phaseTwoFracFlowOwned = phaseTwoFracFlowOverlap;

      const int *neighPtr = localNeighborList;
      double cellVolume;
      ScalarT permeabilityTrace;
      ScalarT phaseOnePorePerm, phaseTwoPorePerm;
      ScalarT dPorePressure, dFracPressure, dFracMinusPorePress, Y_dx, Y_dy, Y_dz, dY, fracPermeability;
      ScalarT phaseOneFracPerm, phaseTwoFracPerm, phaseOneRelPermPores, fracWidth;
      ScalarT phaseOneRelPermFrac, phaseTwoRelPermPores, phaseTwoRelPermFrac, satStarPores, satStarFrac;
      ScalarT scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, scalarPhaseTwoPoreFlow, scalarPhaseTwoFracFlow;
      ScalarT scalarPhaseOneFracToPoreFlow, scalarPhaseTwoFracToPoreFlow, omegaPores, omegaFrac;

      for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                        phaseOneSaturationPoresYOwned++, phaseOneSaturationPoresVOwned++,
                                        phaseOneSaturationFracYOwned++, phaseOneSaturationFracVOwned++,
                                        yOwned +=3, phaseOnePoreFlowOwned++, phaseOneFracFlowOwned++,
                                        phaseTwoPoreFlowOwned++, phaseTwoFracFlowOwned++, deltaT++, damageOwned++,
                                        matrixPorosityNP1Owned++,matrixPorosityNOwned++,
                                        fracturePorosityNP1Owned++,fracturePorosityNOwned++,
                                        phaseOneDensityInPoresNP1Owned++,phaseOneDensityInPoresNOwned++,
                                        phaseOneDensityInFractureNP1Owned++,phaseOneDensityInFractureNOwned++,
                                        phaseTwoDensityInPoresNP1Owned++,phaseTwoDensityInPoresNOwned++,
                                        phaseTwoDensityInFractureNP1Owned++,phaseTwoDensityInFractureNOwned++,
                                        thetaBreakless++,porePressureVOwned++, fracturePressureVOwned++){
        int numNeigh = *neighPtr; neighPtr++;
        double selfCellVolume = v[p];
        const ScalarT *Y = yOwned;
        const ScalarT *porePressureY = porePressureYOwned;
        const double *porePressureV = porePressureVOwned;
        const ScalarT *fracturePressureY = fracturePressureYOwned;
        const double *fracturePressureV = fracturePressureVOwned;
        const ScalarT *phaseOneSaturationPoresY = phaseOneSaturationPoresYOwned;
        const ScalarT *phaseOneSaturationFracY = phaseOneSaturationFracYOwned;

        // compute relative permeabilities assuming no damage effect
        satStarPores = (*phaseOneSaturationPoresY - 0.2)/0.6; // means spec one is water
        satStarFrac = (*phaseOneSaturationFracY - 0.2)/0.6;
        phaseOneRelPermPores = pow(satStarPores, 2.0); //Empirical model, exponent is related to the material
        phaseOneRelPermFrac = pow(satStarFrac, 2.0);
        phaseTwoRelPermPores = pow((-satStarPores+1.0),2.0);
        phaseTwoRelPermFrac = pow((-satStarFrac+1.0),2.0);

        // for to calculate Leakoff
        dFracMinusPorePress = *fracturePressureY - *porePressureY;

        // Fracture permeability
        fracWidth = 2.0*m_horizon/3.0*(*fracturePorosityNP1Owned);  //TODO replace m_horizon/3.0 with actual grid spacing
        fracPermeability = fracWidth*fracWidth/12.0;

        for(int n=0;n<numNeigh;n++,neighPtr++){
          int localId = *neighPtr;
          cellVolume = v[localId];
          const ScalarT *porePressureYP = &porePressureYOverlap[localId];
          const ScalarT *fracturePressureYP = &fracturePressureYOverlap[localId];
          const ScalarT *YP = &yOverlap[3*localId];
          const double *damageNeighbor = &damage[localId]; //TODO synchronize neighbor damage before force evaluation (this is aparently expensive though)

          Y_dx = *(YP+0) - *(Y+0);
          Y_dy = *(YP+1) - *(Y+1);
          Y_dz = *(YP+2) - *(Y+2);
          dY = sqrt(Y_dx*Y_dx+Y_dy*Y_dy+Y_dz*Y_dz);
          //NOTE I want to use DFad<double>, which is why I circumvent the standard influence function code.
          omegaPores = 1.0 - abs(dY/m_horizon);
          //Frac diffusion is a more local process than pore diffusion.
          omegaFrac = 1.0 - abs(dY/m_horizon_fracture);

          // Pressure potential
          dPorePressure = *porePressureYP - *porePressureY;
          dFracPressure = *fracturePressureYP - *fracturePressureY;

          /*
            Nonlocal permeability istropic tensor evaluation result
          */
          permeabilityTrace = (m_matrixPermeabilityXX + m_matrixPermeabilityYY + m_matrixPermeabilityZZ);
          ScalarT m_permeabilityScalar = (m_matrixPermeabilityXX - 0.25 * permeabilityTrace) * Y_dx * Y_dx
                               + (m_matrixPermeabilityYY - 0.25 * permeabilityTrace) * Y_dy * Y_dy
                               + (m_matrixPermeabilityZZ - 0.25 * permeabilityTrace) * Y_dz * Y_dz;

          // Pore permeability is affected by an ad-hoc S-curve relation to damage.
          phaseOnePorePerm = m_permeabilityScalar*phaseOneRelPermPores;
          phaseTwoPorePerm = m_permeabilityScalar*phaseTwoRelPermPores;
          // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
          phaseOneFracPerm = fracPermeability*phaseOneRelPermFrac;
          phaseTwoFracPerm = fracPermeability*phaseTwoRelPermFrac;

          // compute flow density
          const double CORR_FACTOR_FRACTURE = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon_fracture*m_horizon_fracture*m_horizon_fracture);
          const double CORR_FACTOR_PORES = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon*m_horizon*m_horizon);

          // flow entering cell is positive
          scalarPhaseOnePoreFlow = omegaPores * CORR_FACTOR_PORES * (*phaseOneDensityInPoresNP1Owned) / m_phaseOneViscosity * phaseOnePorePerm / pow(dY, 4.0) * dPorePressure;
          scalarPhaseOneFracFlow = omegaFrac * CORR_FACTOR_FRACTURE * (*phaseOneDensityInFractureNP1Owned) / (2.0 * m_phaseOneViscosity) * phaseOneFracPerm / pow(dY, 2.0) * dFracPressure;
          scalarPhaseTwoPoreFlow = omegaPores * CORR_FACTOR_PORES * (*phaseTwoDensityInPoresNP1Owned) / m_phaseTwoViscosity * phaseTwoPorePerm / pow(dY, 4.0) * dPorePressure;
          scalarPhaseTwoFracFlow = omegaFrac * CORR_FACTOR_FRACTURE * (*phaseTwoDensityInFractureNP1Owned) / (2.0 * m_phaseTwoViscosity) * phaseTwoFracPerm / pow(dY, 2.0) * dFracPressure;

          // convert flow density to flow and account for reactions
          *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
          *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
          *phaseTwoPoreFlowOwned += scalarPhaseTwoPoreFlow*cellVolume;
          *phaseTwoFracFlowOwned += scalarPhaseTwoFracFlow*cellVolume;
          phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
          phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
          phaseTwoPoreFlowOverlap[localId] -= scalarPhaseTwoPoreFlow*selfCellVolume;
          phaseTwoFracFlowOverlap[localId] -= scalarPhaseTwoFracFlow*selfCellVolume;
        }

        double permeabilityAvg = (m_matrixPermeabilityXX + m_matrixPermeabilityYY + m_matrixPermeabilityZZ)/3.0;
        //Add in viscous and leakoff terms from mass conservation equation
        //NOTE it is assumed that grid spacing is 1/3 of the standard horizon in order to compute leakoff diffusion area
        *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned -((*phaseOneDensityInPoresNP1Owned)*(*matrixPorosityNP1) - (*phaseOneDensityInPoresNOwned)*(*matrixPorosityN))/deltaTime + permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaBreakless)*m_phaseOneViscosity*(m_horizon/6.0));
        *phaseOneFracFlowOwned = *phaseOneFracFlowOwned -((*phaseOneDensityInFractureNP1Owned)*(*fracturePorosityNP1) - (*phaseOneDensityInFractureNOwned)*(*fracturePorosityN))/deltaTime - permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaBreakless)*m_phaseOneViscosity*(m_horizon/6.0));
        *phaseTwoPoreFlowOwned = *phaseTwoPoreFlowOwned -((*phaseTwoDensityInPoresNP1Owned)*(*matrixPorosityNP1) - (*phaseTwoDensityInPoresNOwned)*(*matrixPorosityN))/deltaTime + permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaBreakless)*m_phaseTwoViscosity*(m_horizon/6.0));
        *phaseTwoFracFlowOwned = *phaseTwoFracFlowOwned -((*phaseTwoDensityInFractureNP1Owned)*(*fracturePorosityNP1) - (*phaseTwoDensityInFractureNOwned)*(*fracturePorosityN))/deltaTime - permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaBreakless)*m_phaseTwoViscosity*(m_horizon/6.0));

      }
    }

    /** Explicit template instantiation for double. */
    template void computeInternalFlow<double>
    (
      const double* yOverlap,
      const double* porePressureYOverlap,
      const double* porePressureVOverlap,
      const double* fracturePressureYOverlap,
      const double* fracturePressureVOverlap,
      const double* phaseOneSaturationPoresYOverlap,
      const double* phaseOneSaturationPoresVOverlap,
      const double* phaseOneSaturationFracYOverlap,
      const double* phaseOneSaturationFracVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const double* matrixPorosityNP1,
      const double* matrixPorosityN,
      const double* fracturePorosityNP1,
      const double* fracturePorosityN,
      const double* phaseOneDensityInPoresNP1,
      const double* phaseOneDensityInPoresN,
      const double* phaseOneDensityInFractureNP1,
      const double* phaseOneDensityInFractureN,
      const double* phaseTwoDensityInPoresNP1,
      const double* phaseTwoDensityInPoresN,
      const double* phaseTwoDensityInFractureNP1,
      const double* phaseTwoDensityInFractureN,
      const double* breaklessDilatationOwned,
      double* phaseOnePoreFlowOverlap,
      double* phaseOneFracFlowOverlap,
      double* phaseTwoPoreFlowOverlap,
      double* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature
    );

    /*Because the complex case needs a different argument pattern we need something
    different than templates. */
    void computeInternalFlowComplex
    (
      const std::complex<double>* yOverlap,
      const std::complex<double>* porePressureYOverlap,
      const std::complex<double>* porePressureVOverlap,
      const std::complex<double>* fracturePressureYOverlap,
      const std::complex<double>* fracturePressureVOverlap,
      const std::complex<double>* phaseOneSaturationPoresYOverlap,
      const std::complex<double>* phaseOneSaturationPoresVOverlap,
      const std::complex<double>* phaseOneSaturationFracYOverlap,
      const std::complex<double>* phaseOneSaturationFracVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const std::complex<double>* matrixPorosityNP1,
      const double* matrixPorosityN,
      const std::complex<double>* fracturePorosityNP1,
      const double* fracturePorosityN,
      const std::complex<double>* phaseOneDensityInPoresNP1,
      const double* phaseOneDensityInPoresN,
      const std::complex<double>* phaseOneDensityInFractureNP1,
      const double* phaseOneDensityInFractureN,
      const std::complex<double>* phaseTwoDensityInPoresNP1,
      const double* phaseTwoDensityInPoresN,
      const std::complex<double>* phaseTwoDensityInFractureNP1,
      const double* phaseTwoDensityInFractureN,
      const std::complex<double>* breaklessDilatationOwned,
      std::complex<double>* phaseOnePoreFlowOverlap,
      std::complex<double>* phaseOneFracFlowOverlap,
      std::complex<double>* phaseTwoPoreFlowOverlap,
      std::complex<double>* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_matrixPermeabilityXX,
      const double m_matrixPermeabilityYY,
      const double m_matrixPermeabilityZZ,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double deltaTime,
      const double* deltaTemperature
    )
    {
      /*
       * Compute processor local contribution to internal fluid flow
       */

      const std::complex<double> *yOwned = yOverlap;
      const std::complex<double> *porePressureVOwned = porePressureVOverlap;
      const std::complex<double> *porePressureYOwned = porePressureYOverlap;
      const std::complex<double> *fracturePressureVOwned = fracturePressureVOverlap;
      const std::complex<double> *fracturePressureYOwned = fracturePressureYOverlap;
      const std::complex<double> *phaseOneSaturationPoresYOwned = phaseOneSaturationPoresYOverlap;
      const std::complex<double> *phaseOneSaturationPoresVOwned = phaseOneSaturationPoresVOverlap;
      const std::complex<double> *phaseOneSaturationFracYOwned = phaseOneSaturationFracYOverlap;
      const std::complex<double> *phaseOneSaturationFracVOwned = phaseOneSaturationFracVOverlap;

      const double *v = volumeOverlap;
      const double *deltaT = deltaTemperature;
      const double *damageOwned = damage;

      const std::complex<double>* matrixPorosityNP1Owned = matrixPorosityNP1;
      const double* matrixPorosityNOwned = matrixPorosityN;
      const std::complex<double>* fracturePorosityNP1Owned = fracturePorosityNP1;
      const double* fracturePorosityNOwned= fracturePorosityN;

      const std::complex<double>* phaseOneDensityInPoresNP1Owned = phaseOneDensityInPoresNP1;
      const double* phaseOneDensityInPoresNOwned = phaseOneDensityInPoresN;
      const std::complex<double>* phaseOneDensityInFractureNP1Owned = phaseOneDensityInFractureNP1;
      const double* phaseOneDensityInFractureNOwned = phaseOneDensityInFractureN;

      const std::complex<double>* phaseTwoDensityInPoresNP1Owned = phaseTwoDensityInPoresNP1;
      const double* phaseTwoDensityInPoresNOwned = phaseTwoDensityInPoresN;
      const std::complex<double>* phaseTwoDensityInFractureNP1Owned = phaseTwoDensityInFractureNP1;
      const double* phaseTwoDensityInFractureNOwned = phaseTwoDensityInFractureN;

      const std::complex<double> *thetaBreakless = breaklessDilatationOwned;

      std::complex<double> *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
      std::complex<double> *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;
      std::complex<double> *phaseTwoPoreFlowOwned = phaseTwoPoreFlowOverlap;
      std::complex<double> *phaseTwoFracFlowOwned = phaseTwoFracFlowOverlap;

      const int *neighPtr = localNeighborList;
      double cellVolume;
      std::complex<double> phaseOnePorePerm, phaseTwoPorePerm, permeabilityTrace;
      std::complex<double> dPorePressure, dFracPressure, dFracMinusPorePress, Y_dx, Y_dy, Y_dz, dY, fracPermeability;
      std::complex<double> phaseOneFracPerm, phaseTwoFracPerm, phaseOneRelPermPores, fracWidth;
      std::complex<double> phaseOneRelPermFrac, phaseTwoRelPermPores, phaseTwoRelPermFrac, satStarPores, satStarFrac;
      std::complex<double> scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, scalarPhaseTwoPoreFlow, scalarPhaseTwoFracFlow;
      std::complex<double> scalarPhaseOneFracToPoreFlow, scalarPhaseTwoFracToPoreFlow, omegaPores, omegaFrac;

      for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                        phaseOneSaturationPoresYOwned++, phaseOneSaturationPoresVOwned++,
                                        phaseOneSaturationFracYOwned++, phaseOneSaturationFracVOwned++,
                                        yOwned +=3, phaseOnePoreFlowOwned++, phaseOneFracFlowOwned++,
                                        phaseTwoPoreFlowOwned++, phaseTwoFracFlowOwned++, deltaT++, damageOwned++,
                                        matrixPorosityNP1Owned++,matrixPorosityNOwned++,
                                        fracturePorosityNP1Owned++,fracturePorosityNOwned++,
                                        phaseOneDensityInPoresNP1Owned++,phaseOneDensityInPoresNOwned++,
                                        phaseOneDensityInFractureNP1Owned++,phaseOneDensityInFractureNOwned++,
                                        phaseTwoDensityInPoresNP1Owned++,phaseTwoDensityInPoresNOwned++,
                                        phaseTwoDensityInFractureNP1Owned++,phaseTwoDensityInFractureNOwned++,
                                        thetaBreakless++,porePressureVOwned++, fracturePressureVOwned++){
        int numNeigh = *neighPtr; neighPtr++;
        double selfCellVolume = v[p];
        const std::complex<double> *Y = yOwned;
        const std::complex<double> *porePressureY = porePressureYOwned;
        const std::complex<double> *porePressureV = porePressureVOwned;
        const std::complex<double> *fracturePressureY = fracturePressureYOwned;
        const std::complex<double> *fracturePressureV = fracturePressureVOwned;
        const std::complex<double> *phaseOneSaturationPoresY = phaseOneSaturationPoresYOwned;
        const std::complex<double> *phaseOneSaturationFracY = phaseOneSaturationFracYOwned;

        // compute relative permeabilities assuming no damage effect
        satStarPores = (*phaseOneSaturationPoresY - 0.2)/0.6; // means spec one is water
        satStarFrac = (*phaseOneSaturationFracY - 0.2)/0.6;
        phaseOneRelPermPores = pow(satStarPores, 2.0); //Empirical model, exponent is related to the material
        phaseOneRelPermFrac = pow(satStarFrac, 2.0);
        phaseTwoRelPermPores = pow((-satStarPores+1.0),2.0);
        phaseTwoRelPermFrac = pow((-satStarFrac+1.0),2.0);

        // for to calculate Leakoff
        dFracMinusPorePress = *fracturePressureY - *porePressureY;

        // Fracture permeability
        fracWidth = 2.0*m_horizon/3.0*(*fracturePorosityNP1Owned);  //TODO replace m_horizon/3.0 with actual grid spacing
        fracPermeability = fracWidth*fracWidth/12.0;

        for(int n=0;n<numNeigh;n++,neighPtr++){
          int localId = *neighPtr;
          cellVolume = v[localId];
          const std::complex<double> *porePressureYP = &porePressureYOverlap[localId];
          const std::complex<double> *fracturePressureYP = &fracturePressureYOverlap[localId];
          const std::complex<double> *YP = &yOverlap[3*localId];
          const double *damageNeighbor = &damage[localId]; //TODO synchronize neighbor damage before force evaluation (this is aparently expensive though)

          Y_dx = *(YP+0) - *(Y+0);
          Y_dy = *(YP+1) - *(Y+1);
          Y_dz = *(YP+2) - *(Y+2);
          dY = sqrt(Y_dx*Y_dx+Y_dy*Y_dy+Y_dz*Y_dz);
          //NOTE I want to use std::complex<double>, which is why I circumvent the standard influence function code.
          //NOTE real part needs to be nonnegative.
          omegaPores =  std::complex<double>(1.0, 0.0) - std::abs(std::real(dY/m_horizon)) - std::complex<double>(0.0, std::imag(dY/m_horizon));
          //Frac diffusion is a more local process than pore diffusion.
          omegaFrac =  std::complex<double>(1.0, 0.0) - std::abs(std::real(dY/m_horizon_fracture)) - std::complex<double>(0.0, std::imag(dY/m_horizon_fracture));

          // Pressure potential
          dPorePressure = *porePressureYP - *porePressureY;
          dFracPressure = *fracturePressureYP - *fracturePressureY;

          /*
            Nonlocal permeability istropic tensor evaluation result
          */
          permeabilityTrace = (m_matrixPermeabilityXX + m_matrixPermeabilityYY + m_matrixPermeabilityZZ);
          const std::complex<double> m_permeabilityScalar = (m_matrixPermeabilityXX - 0.25 * permeabilityTrace) * Y_dx * Y_dx
                               + (m_matrixPermeabilityYY - 0.25 * permeabilityTrace) * Y_dy * Y_dy
                               + (m_matrixPermeabilityZZ - 0.25 * permeabilityTrace) * Y_dz * Y_dz;

          // Pore permeability is affected by an ad-hoc S-curve relation to damage.
          phaseOnePorePerm = m_permeabilityScalar*phaseOneRelPermPores;
          phaseTwoPorePerm = m_permeabilityScalar*phaseTwoRelPermPores;
          // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
          phaseOneFracPerm = fracPermeability*phaseOneRelPermFrac;
          phaseTwoFracPerm = fracPermeability*phaseTwoRelPermFrac;

          // compute flow density
          const double CORR_FACTOR_FRACTURE = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon_fracture*m_horizon_fracture*m_horizon_fracture);
          const double CORR_FACTOR_PORES = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon*m_horizon*m_horizon);

          // flow entering cell is positive
          scalarPhaseOnePoreFlow = omegaPores * CORR_FACTOR_PORES * (*phaseOneDensityInPoresNP1Owned) / m_phaseOneViscosity * phaseOnePorePerm / pow(dY, 4.0) * dPorePressure;
          scalarPhaseOneFracFlow = omegaFrac * CORR_FACTOR_FRACTURE * (*phaseOneDensityInFractureNP1Owned) / (2.0 * m_phaseOneViscosity) * phaseOneFracPerm / pow(dY, 2.0) * dFracPressure;
          scalarPhaseTwoPoreFlow = omegaPores * CORR_FACTOR_PORES * (*phaseTwoDensityInPoresNP1Owned) / m_phaseTwoViscosity * phaseTwoPorePerm / pow(dY, 4.0) * dPorePressure;
          scalarPhaseTwoFracFlow = omegaFrac * CORR_FACTOR_FRACTURE * (*phaseTwoDensityInFractureNP1Owned) / (2.0 * m_phaseTwoViscosity) * phaseTwoFracPerm / pow(dY, 2.0) * dFracPressure;

          // convert flow density to flow and account for reactions
          *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
          *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
          *phaseTwoPoreFlowOwned += scalarPhaseTwoPoreFlow*cellVolume;
          *phaseTwoFracFlowOwned += scalarPhaseTwoFracFlow*cellVolume;
          phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
          phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
          phaseTwoPoreFlowOverlap[localId] -= scalarPhaseTwoPoreFlow*selfCellVolume;
          phaseTwoFracFlowOverlap[localId] -= scalarPhaseTwoFracFlow*selfCellVolume;
        }

        double permeabilityAvg = (m_matrixPermeabilityXX + m_matrixPermeabilityYY + m_matrixPermeabilityZZ)/3.0;
        //Add in viscous and leakoff terms from mass conservation equation
        //NOTE it is assumed that grid spacing is 1/3 of the standard horizon in order to compute leakoff diffusion area
        *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned -((*phaseOneDensityInPoresNP1Owned)*(*matrixPorosityNP1) - (*phaseOneDensityInPoresNOwned)*(*matrixPorosityN))/deltaTime + permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaBreakless)*m_phaseOneViscosity*(m_horizon/6.0));
        *phaseOneFracFlowOwned = *phaseOneFracFlowOwned -((*phaseOneDensityInFractureNP1Owned)*(*fracturePorosityNP1) - (*phaseOneDensityInFractureNOwned)*(*fracturePorosityN))/deltaTime - permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaBreakless)*m_phaseOneViscosity*(m_horizon/6.0));
        *phaseTwoPoreFlowOwned = *phaseTwoPoreFlowOwned -((*phaseTwoDensityInPoresNP1Owned)*(*matrixPorosityNP1) - (*phaseTwoDensityInPoresNOwned)*(*matrixPorosityN))/deltaTime + permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaBreakless)*m_phaseTwoViscosity*(m_horizon/6.0));
        *phaseTwoFracFlowOwned = *phaseTwoFracFlowOwned -((*phaseTwoDensityInFractureNP1Owned)*(*fracturePorosityNP1) - (*phaseTwoDensityInFractureNOwned)*(*fracturePorosityN))/deltaTime - permeabilityAvg*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaBreakless)*m_phaseTwoViscosity*(m_horizon/6.0));

      }
    }
  }

}
