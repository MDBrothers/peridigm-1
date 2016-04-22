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
      double pressureInMPa = (*pressureOwned)*1.0e-6;
      double pressInMPaSquared = pressureInMPa*pressureInMPa;
      double tempSquared = (*temperatureOwned)*(*temperatureOwned);

      // Empirical relation supplied to the developer by Ouichi Hisanao
  	  *densityOwned = (-0.00000014569010515*pressInMPaSquared + 0.000046724532297*pressureInMPa - 0.0061488874609)*tempSquared
  		+ (0.000088493144499*pressInMPaSquared - 0.029002566308*pressureInMPa + 3.3982146161)*Temperature
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
      double pressureInMPa = (*pressureOwned)*1.0e-6;
      double pressInMPaSquared = pressureInMPa*pressureInMPa;
      double tempSquared = (*temperatureOwned)*(*temperatureOwned);

      // Empirical relation supplied to the developer by Ouichi Hisanao
  	  *densityOwned = (-0.00000014569010515*pressInMPaSquared + 0.000046724532297*pressureInMPa - 0.0061488874609)*tempSquared
  		+ (0.000088493144499*pressInMPaSquared - 0.029002566308*pressureInMPa + 3.3982146161)*Temperature
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
      const double* principleDamageDirection,
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
      const double m_permeabilityScalar,
      const double m_phaseOneDensity,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
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
      const double *principleDamageDirectionOwned = principleDamageDirection;

      const ScalarT *matrixPorosityOwnedNP1 = matrixPorosityNP1;
      const double *matrixPorosityOwnedN = matrixPorosityN;
      const ScalarT *fracturePorosityOwnedNP1 = fracturePorosityNP1;
      const double *fracturePorosityOwnedN = fracturePorosityN;

    	const ScalarT* phaseOneDensityInPoresOwnedNP1 = phaseOneDensityInPoresNP1;
    	const double* phaseOneDensityInPoresOwnedN = phaseOneDensityInPoresN;
    	const ScalarT* phaseOneDensityInFractureOwnedNP1 = phaseOneDensityInFractureNP1;
    	const double* phaseOneDensityInFractureOwnedN = phaseOneDensityInFractureN;

      const ScalarT *thetaLocal = breaklessDilatation;
      ScalarT *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
      ScalarT *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;

      const int *neighPtr = localNeighborList;
      double cellVolume, harmonicAverageDamage;
      ScalarT phaseOnePorePerm,  dPorePressure, dFracPressure;
      ScalarT dFracMinusPorePress, Y_dx, Y_dy, Y_dz, dY, fracWidth, fracPermeability;             // SA: fracWidth introduced
      ScalarT fractureDirectionFactor, phaseOneFracPerm;
      ScalarT scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow;
      ScalarT scalarPhaseOneFracToPoreFlow, omegaPores, omegaFrac;

      for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                        yOwned +=3, phaseOnePoreFlowOwned++,
                                        phaseOneFracFlowOwned++, deltaT++,
                                        damageOwned++, thetaLocal++,
                                        principleDamageDirectionOwned +=3,
                                        matrixPorosityOwnedNP1++, fracturePorosityOwnedNP1++,
                                        matrixPorosityOwnedN++, fracturePorosityOwnedN++,
    																		phaseOneDensityInFractureOwnedNP1++, phaseOneDensityInPoresOwnedNP1++,
    																		phaseOneDensityInFractureOwnedN++, phaseOneDensityInPoresOwnedN++,
                                        porePressureVOwned++, fracturePressureVOwned++){
        int numNeigh = *neighPtr; neighPtr++;
        double selfCellVolume = v[p];
        const ScalarT *Y = yOwned;
        const ScalarT *porePressureY = porePressureYOwned;
        const double *porePressureV = porePressureVOwned;
        const ScalarT *fracturePressureY = fracturePressureYOwned;
        const double *fracturePressureV = fracturePressureVOwned;
        const double *principleDamageDirection = principleDamageDirectionOwned;

        // Fracture permeability
        fracWidth = 2.0*m_horizon*(*fracturePorosityOwnedNP1);                   //TODO change this to grid spacing from m_horizon
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
          //NOTE I want to use DFad<double>, which is why I circumvent the standard influence function code.
          omegaPores = exp(-dY*dY/(m_horizon*m_horizon));// scalarInfluenceFunction(dY,m_horizon);
          //Frac diffusion is a more local process than pore diffusion.
          omegaFrac = exp(-dY*dY/(m_horizon_fracture*m_horizon_fracture));// scalarInfluenceFunction(dY,m_horizon_fracture);

          // Pressure potential
          dPorePressure = *porePressureYP - *porePressureY;
          dFracPressure = *fracturePressureYP - *fracturePressureY;

          // compute permeabilities
          // Frac permeability in directions other than orthogonal to the principle damage direction is strongly attenuated.
          //fractureDirectionFactor = pow(cos(Y_dx*(*(principleDamageDirection+0)) + Y_dy*(*(principleDamageDirection+1)) + Y_dz*(*(principleDamageDirection+2))),2.0); //Frac flow allowed in direction perpendicular to damage
          // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
          phaseOneFracPerm = fracPermeability;//*fractureDirectionFactor;

          /*
            Nonlocal permeability istropic tensor evaluation result
          */
          phaseOnePorePerm = dY*dY*m_permeabilityScalar/4.0;

          const double CORR_FACTOR_FRACTURE = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon_fracture*m_horizon_fracture*m_horizon_fracture);
          const double CORR_FACTOR_PORES = 45.0/(4.0*boost::math::constants::pi<double>()*m_horizon*m_horizon*m_horizon);

          // compute flow density
          // flow entering cell is positive
          scalarPhaseOnePoreFlow = omegaPores * CORR_FACTOR_PORES * (*phaseOneDensityInPoresOwnedNP1) / m_phaseOneViscosity * m_permeabilityScalar / pow(dY, 4.0) * dPorePressure;
          scalarPhaseOneFracFlow = omegaFrac * CORR_FACTOR_FRACTURE * (*phaseOneDensityInFractureOwnedNP1) / (2.0 * m_phaseOneViscosity) * phaseOneFracPerm / pow(dY, 2.0) * dFracPressure;

          // convert flow density to flow and account for reactions
          *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
          *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
          phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
          phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
        }

        //Add in viscous and leakoff terms from mass conservation equation
        *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned -((*phaseOneDensityInPoresOwnedNP1)*(*matrixPorosityNP1) - (*phaseOneDensityInPoresOwnedN)*(*matrixPorosityN))/deltaTime + m_permeabilityScalar*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/2.0));
        *phaseOneFracFlowOwned = *phaseOneFracFlowOwned -((*phaseOneDensityInFractureOwnedNP1)*(*fracturePorosityNP1) - (*phaseOneDensityInFractureOwnedN)*(*fracturePorosityN))/deltaTime - m_permeabilityScalar*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/2.0));
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
      const double* principleDamageDirection,
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
      const double m_permeabilityScalar,
      const double m_phaseOneDensity,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double* deltaTemperature
    );

    /** Explicit template instantiation for Sacado::Fad::DFad<double>. */
    template void computeInternalFlow<Sacado::Fad::DFad<double> >
    (
      const Sacado::Fad::DFad<double> * yOverlap,
      const Sacado::Fad::DFad<double> * porePressureYOverlap,
      const double* porePressureVOverlap,
      const Sacado::Fad::DFad<double> * fracturePressureYOverlap,
      const double* fracturePressureVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const double* principleDamageDirection,
      const Sacado::Fad::DFad<double> * matrixPorosityNP1,
      const double* matrixPorosityN,
      const Sacado::Fad::DFad<double> * fracturePorosityNP1,
      const double* fracturePorosityN,
    	const Sacado::Fad::DFad<double> * phaseOneDensityInPoresNP1,
    	const double* phaseOneDensityInPoresN,
    	const Sacado::Fad::DFad<double> * phaseOneDensityInFractureNP1,
    	const double * phaseOneDensityInFractureN,
      const Sacado::Fad::DFad<double> * breaklessDilatation,
      Sacado::Fad::DFad<double> * phaseOnePoreFlowOverlap,
      Sacado::Fad::DFad<double> * phaseOneFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_permeabilityScalar,
      const double m_phaseOneDensity,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double* deltaTemperature
    );

    template void computeInternalFlowComplex
    (
      const std::complex<double> * yOverlap,
      const std::complex<double> * porePressureYOverlap,
      const std::complex<double> * porePressureVOverlap,
      const std::complex<double> * fracturePressureYOverlap,
      const std::complex<double> * fracturePressureVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const double* principleDamageDirection,
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
      const double m_permeabilityScalar,
      const double m_phaseOneDensity,
      const double m_phaseOneViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double* deltaTemperature
    )
    {
      /*
       * Compute processor local contribution to internal fluid flow
       */
      const std::complex<double> *yOwned = yOverlap;
      const double *porePressureVOwned = porePressureVOverlap;
      const std::complex<double> *porePressureYOwned = porePressureYOverlap;
      const double *fracturePressureVOwned = fracturePressureVOverlap;
      const std::complex<double> *fracturePressureYOwned = fracturePressureYOverlap;

      const double *v = volumeOverlap;
      const double *deltaT = deltaTemperature;
      const double *damageOwned = damage;
      const double *principleDamageDirectionOwned = principleDamageDirection;

      const std::complex<double> *matrixPorosityOwnedNP1 = matrixPorosityNP1;
      const double *matrixPorosityOwnedN = matrixPorosityN;
      const std::complex<double> *fracturePorosityOwnedNP1 = fracturePorosityNP1;
      const double *fracturePorosityOwnedN = fracturePorosityN;

      const std::complex<double>* phaseOneDensityInPoresOwnedNP1 = phaseOneDensityInPoresNP1;
      const double* phaseOneDensityInPoresOwnedN = phaseOneDensityInPoresN;
      const std::complex<double>* phaseOneDensityInFractureOwnedNP1 = phaseOneDensityInFractureNP1;
      const double* phaseOneDensityInFractureOwnedN = phaseOneDensityInFractureN;

      const std::complex<double> *thetaLocal = breaklessDilatation;
      std::complex<double> *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
      std::complex<double> *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;

      const int *neighPtr = localNeighborList;
      double cellVolume, harmonicAverageDamage;
      std::complex<double> phaseOnePorePerm,  dPorePressure, dFracPressure;
      std::complex<double> dFracMinusPorePress, Y_dx, Y_dy, Y_dz, dY, fracPermeability;
      std::complex<double> fractureDirectionFactor, phaseOneFracPerm, phaseOneRelPermPores,
      std::complex<double> scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, phaseOneRelPermFrac;
      std::complex<double> scalarPhaseOneFracToPoreFlow, omegaPores, omegaFrac;

      for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                        yOwned +=3, phaseOnePoreFlowOwned++,
                                        phaseOneFracFlowOwned++, deltaT++,
                                        damageOwned++, thetaLocal++,
                                        principleDamageDirectionOwned +=3,
                                        matrixPorosityOwnedNP1++, fracturePorosityOwnedNP1++,
                                        matrixPorosityOwnedN++, fracturePorosityOwnedN++,
                                        phaseOneDensityInFractureOwnedNP1++, phaseOneDensityInPoresOwnedNP1++,
                                        phaseOneDensityInFractureOwnedN++, phaseOneDensityInPoresOwnedN++,
                                        porePressureVOwned++, fracturePressureVOwned++){
        int numNeigh = *neighPtr; neighPtr++;
        double selfCellVolume = v[p];
        const std::complex<double> *Y = yOwned;
        const std::complex<double> *porePressureY = porePressureYOwned;
        const double *porePressureV = porePressureVOwned;
        const std::complex<double> *fracturePressureY = fracturePressureYOwned;
        const double *fracturePressureV = fracturePressureVOwned;
        const double *principleDamageDirection = principleDamageDirectionOwned;

        // Fracture permeability
        fracWidth = 2.0*m_horizon*(*fracturePorosityOwnedNP1);
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
          //NOTE I want to use DFad<double>, which is why I circumvent the standard influence function code.
          omegaPores = exp(-dY*dY/(m_horizon*m_horizon));// scalarInfluenceFunction(dY,m_horizon);
          //Frac diffusion is a more local process than pore diffusion.
          omegaFrac = exp(-dY*dY/(m_horizon_fracture*m_horizon_fracture));// scalarInfluenceFunction(dY,m_horizon_fracture);

          // Pressure potential
          dPorePressure = *porePressureYP - *porePressureY;
          dFracPressure = *fracturePressureYP - *fracturePressureY;

          // compute permeabilities
          // Frac permeability in directions other than orthogonal to the principle damage direction is strongly attenuated.
          fractureDirectionFactor = pow(cos(Y_dx*(*(principleDamageDirection+0)) + Y_dy*(*(principleDamageDirection+1)) + Y_dz*(*(principleDamageDirection+2))),2.0); //Frac flow allowed in direction perpendicular to damage
          // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
          phaseOneFracPerm = fracPermeability*fractureDirectionFactor;

          // compute flow density
          // flow entering cell is positive
          scalarPhaseOnePoreFlow = omegaPores * (*phaseOneDensityInPoresOwnedNP1) / m_phaseOneViscosity * m_permeabilityScalar / pow(dY, 4.0) * dPorePressure;
          scalarPhaseOneFracFlow = omegaFrac * (*phaseOneDensityInFractureOwnedNP1) / (2.0 * m_phaseOneViscosity) * phaseOneFracPerm / pow(dY, 2.0) * dFracPressure;

          // convert flow density to flow and account for reactions
          *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
          *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
          phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
          phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
        }

        //Add in viscous and leakoff terms from mass conservation equation
        *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned -((*phaseOneDensityInPoresOwnedNP1)*(*matrixPorosityNP1) - (*phaseOneDensityInPoresOwnedN)*(*matrixPorosityN))/deltaTime + m_permeabilityScalar*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/2.0));
        *phaseOneFracFlowOwned = *phaseOneFracFlowOwned -((*phaseOneDensityInFractureOwnedNP1)*(*fracturePorosityNP1) - (*phaseOneDensityInFractureOwnedN)*(*fracturePorosityN))/deltaTime - m_permeabilityScalar*4.0*M_PI*m_horizon_fracture*m_horizon_fracture*dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/2.0));
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
      const double* principleDamageDirection,
      const double* criticalDilatationOwned,
      const ScalarT* breaklessDilatationOwned,
      ScalarT* phaseOnePoreFlowOverlap,
      ScalarT* phaseOneFracFlowOverlap,
      ScalarT* phaseTwoPoreFlowOverlap,
      ScalarT* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_permeabilityScalar,
      const double m_permeabilityCurveInflectionDamage,
      const double m_permeabilityAlpha,
      const double m_maxPermeability,
      const double m_phaseOneBasePerm,
      const double m_phaseTwoBasePerm,
      const double m_phaseOneDensity,
      const double m_phaseTwoDensity,
      const double m_phaseOneCompressibility,
      const double m_phaseTwoCompressibility,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
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
      const double *principleDamageDirectionOwned = principleDamageDirection;
      const double *thetaCritical = criticalDilatationOwned;
      const ScalarT *thetaBreakless = breaklessDilatationOwned;

      ScalarT *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
      ScalarT *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;
      ScalarT *phaseTwoPoreFlowOwned = phaseTwoPoreFlowOverlap;
      ScalarT *phaseTwoFracFlowOwned = phaseTwoFracFlowOverlap;

      const int *neighPtr = localNeighborList;
      double cellVolume, harmonicAverageDamage;
      ScalarT phaseOnePorePerm, phaseTwoPorePerm;
      ScalarT dPorePressure, dFracPressure, dLocalPoreFracPressure, Y_dx, Y_dy, Y_dz, dY, fracPermeability;
      ScalarT fractureDirectionFactor, phaseOneFracPerm, phaseTwoFracPerm, phaseOneRelPermPores;
      ScalarT phaseOneRelPermFrac, phaseTwoRelPermPores, phaseTwoRelPermFrac, satStarPores, satStarFrac;
      ScalarT scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, scalarPhaseTwoPoreFlow, scalarPhaseTwoFracFlow;
      ScalarT scalarPhaseOneFracToPoreFlow, scalarPhaseTwoFracToPoreFlow, omegaPores, omegaFrac;

      for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                        phaseOneSaturationPoresYOwned++, phaseOneSaturationPoresVOwned++,
                                        phaseOneSaturationFracYOwned++, phaseOneSaturationFracVOwned++,
                                        yOwned +=3, phaseOnePoreFlowOwned++, phaseOneFracFlowOwned++,
                                        phaseTwoPoreFlowOwned++, phaseTwoFracFlowOwned++, deltaT++, damageOwned++,
                                        principleDamageDirectionOwned +=3, thetaCritical++, thetaBreakless++,
                                        porePressureVOwned++, fracturePressureVOwned++){
        int numNeigh = *neighPtr; neighPtr++;
        double selfCellVolume = v[p];
        const ScalarT *Y = yOwned;
        const ScalarT *porePressureY = porePressureYOwned;
        const double *porePressureV = porePressureVOwned;
        const ScalarT *fracturePressureY = fracturePressureYOwned;
        const double *fracturePressureV = fracturePressureVOwned;
        const ScalarT *phaseOneSaturationPoresY = phaseOneSaturationPoresYOwned;
        const ScalarT *phaseOneSaturationFracY = phaseOneSaturationFracYOwned;
        const double *principleDamageDirection = principleDamageDirectionOwned;

        // compute relative permeabilities assuming no damage effect
        satStarPores = (*phaseOneSaturationPoresY - 0.2)/0.6; // means spec one is water
        satStarFrac = (*phaseOneSaturationFracY - 0.2)/0.6;
        phaseOneRelPermPores = m_phaseOneBasePerm*pow(satStarPores, 2.0); //Empirical model, exponent is related to the material
        phaseOneRelPermFrac = m_phaseOneBasePerm*pow(satStarFrac, 2.0);
        phaseTwoRelPermPores = m_phaseTwoBasePerm*pow((-satStarPores+1.0),2.0);
        phaseTwoRelPermFrac = m_phaseTwoBasePerm*pow((-satStarFrac+1.0),2.0);

        // for to calculate Leakoff
        dLocalPoreFracPressure = *fracturePressureY - *porePressureY;

        //compute fracture width based on a two sphere diameter difference. An ad-hoc relation.
        if(*thetaBreakless > 0.0){
          fracPermeability = pow(6.0/M_PI*selfCellVolume*(*thetaBreakless),1.0/3.0) - pow(6.0/M_PI*selfCellVolume*(*thetaCritical),1.0/3.0);
          if(fracPermeability < 0.0)
            fracPermeability = 0.0; //Closed fractures have no flow
        }
        else
          fracPermeability = 0.0;

        fracPermeability *= fracPermeability/12.0; //Empirical relation for permeability

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
          omegaPores = exp(-dY*dY/(m_horizon*m_horizon));// scalarInfluenceFunction(dY,m_horizon);
          //Frac diffusion is a more local process than pore diffusion.
          omegaFrac = exp(-dY*dY/(m_horizon_fracture*m_horizon_fracture));// scalarInfluenceFunction(dY,m_horizon_fracture);

          // Pressure potential
          dPorePressure = *porePressureYP - *porePressureY;
          dFracPressure = *fracturePressureYP - *fracturePressureY;

          // compute permeabilities
          // Frac permeability in directions other than orthogonal to the principle damage direction is strongly attenuated.
          fractureDirectionFactor = pow(cos(Y_dx*(*(principleDamageDirection+0)) + Y_dy*(*(principleDamageDirection+1)) + Y_dz*(*(principleDamageDirection+2))),2.0); //Frac flow allowed in direction perpendicular to damage
          harmonicAverageDamage = 1.0 / (1.0 / *damageOwned + 1.0 / *damageNeighbor);
          if(harmonicAverageDamage != harmonicAverageDamage) harmonicAverageDamage=0.0; //test for nan which occurs when a damage is zero.
          // Pore permeability is affected by an ad-hoc S-curve relation to damage.
          phaseOnePorePerm = m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(exp(-m_permeabilityAlpha*(harmonicAverageDamage - m_permeabilityCurveInflectionDamage))+1.0);
          phaseTwoPorePerm = m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(exp(-m_permeabilityAlpha*(harmonicAverageDamage - m_permeabilityCurveInflectionDamage))+1.0);
          // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
          phaseOneFracPerm = fracPermeability*fractureDirectionFactor*phaseOneRelPermFrac;
          phaseTwoFracPerm = fracPermeability*fractureDirectionFactor*phaseTwoRelPermFrac;

          // compute flow density
          // flow entering cell is positive
          scalarPhaseOnePoreFlow = omegaPores * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseOnePorePerm / pow(dY, 4.0)) * dPorePressure;
          scalarPhaseOneFracFlow = omegaFrac * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseOneFracPerm / pow(dY, 4.0)) * dFracPressure;
          scalarPhaseTwoPoreFlow = omegaPores * m_phaseTwoDensity / m_phaseTwoViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseTwoPorePerm / pow(dY, 4.0)) * dPorePressure;
          scalarPhaseTwoFracFlow = omegaFrac * m_phaseTwoDensity / m_phaseTwoViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseTwoFracPerm / pow(dY, 4.0)) * dFracPressure;

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

        //Leakoff calculation, self to self
        *phaseOnePoreFlowOwned += m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
        *phaseOneFracFlowOwned -= m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
        *phaseTwoPoreFlowOwned += m_phaseTwoDensity / m_phaseTwoViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
        *phaseTwoFracFlowOwned -= m_phaseTwoDensity / m_phaseTwoViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;

        //Viscous terms
        double Porosity= 0.2;
        double B_formation_vol_factor_water = 1.0;
        double B_formation_vol_factor_oil = 1.0;
        double Compressibility_rock = 1.0;
        *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned - *porePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity*(*phaseOneSaturationPoresYOwned)/B_formation_vol_factor_water + Porosity/B_formation_vol_factor_water*(*phaseOneSaturationPoresVOwned);
        *phaseOneFracFlowOwned = *phaseOneFracFlowOwned - *fracturePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity*(*phaseOneSaturationFracYOwned)/B_formation_vol_factor_water + Porosity/B_formation_vol_factor_water*(*phaseOneSaturationFracVOwned);
        *phaseTwoPoreFlowOwned = *phaseTwoPoreFlowOwned - *porePressureV*m_phaseTwoDensity*(Compressibility_rock + m_phaseTwoCompressibility)*Porosity*(1.0 - *phaseOneSaturationPoresYOwned)/B_formation_vol_factor_oil + Porosity/B_formation_vol_factor_oil*(*phaseOneSaturationPoresVOwned);
        *phaseTwoFracFlowOwned = *phaseTwoFracFlowOwned - *fracturePressureV*m_phaseTwoDensity*(Compressibility_rock + m_phaseTwoCompressibility)*Porosity*(1.0 - *phaseOneSaturationFracYOwned)/B_formation_vol_factor_oil + Porosity/B_formation_vol_factor_oil*(*phaseOneSaturationFracVOwned);
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
      const double* principleDamageDirection,
      const double* criticalDilatationOwned,
      const double* breaklessDilatationOwned,
      double* phaseOnePoreFlowOverlap,
      double* phaseOneFracFlowOverlap,
      double* phaseTwoPoreFlowOverlap,
      double* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_permeabilityScalar,
      const double m_permeabilityCurveInflectionDamage,
      const double m_permeabilityAlpha,
      const double m_maxPermeability,
      const double m_phaseOneBasePerm,
      const double m_phaseTwoBasePerm,
      const double m_phaseOneDensity,
      const double m_phaseTwoDensity,
      const double m_phaseOneCompressibility,
      const double m_phaseTwoCompressibility,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double* deltaTemperature
    );

    /** Explicit template instantiation for Sacado::Fad::DFad<double>. */
    template void computeInternalFlow<Sacado::Fad::DFad<double> >
    (
    	const Sacado::Fad::DFad<double>* yOverlap,
      const Sacado::Fad::DFad<double>* porePressureYOverlap,
      const double* porePressureVOverlap,
      const Sacado::Fad::DFad<double>* fracturePressureYOverlap,
      const double* fracturePressureVOverlap,
      const Sacado::Fad::DFad<double>* phaseOneSaturationPoresYOverlap,
      const double* phaseOneSaturationPoresVOverlap,
      const Sacado::Fad::DFad<double>* phaseOneSaturationFracYOverlap,
      const double* phaseOneSaturationFracVOverlap,
      const double* volumeOverlap,
      const double* damage,
      const double* principleDamageDirection,
      const double* criticalDilatationOwned,
      const Sacado::Fad::DFad<double>* breaklessDilatationOwned,
      Sacado::Fad::DFad<double>* phaseOnePoreFlowOverlap,
      Sacado::Fad::DFad<double>* phaseOneFracFlowOverlap,
      Sacado::Fad::DFad<double>* phaseTwoPoreFlowOverlap,
      Sacado::Fad::DFad<double>* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_permeabilityScalar,
      const double m_permeabilityCurveInflectionDamage,
      const double m_permeabilityAlpha,
      const double m_maxPermeability,
      const double m_phaseOneBasePerm,
      const double m_phaseTwoBasePerm,
      const double m_phaseOneDensity,
      const double m_phaseTwoDensity,
      const double m_phaseOneCompressibility,
      const double m_phaseTwoCompressibility,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
      const double* deltaTemperature
    );

    /*Because the complex case needs a difference argument pattern we need something
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
      const double* principleDamageDirection,
      const double* criticalDilatationOwned,
      const std::complex<double>* breaklessDilatationOwned,
      std::complex<double>* phaseOnePoreFlowOverlap,
      std::complex<double>* phaseOneFracFlowOverlap,
      std::complex<double>* phaseTwoPoreFlowOverlap,
      std::complex<double>* phaseTwoFracFlowOverlap,
      const int*  localNeighborList,
      const int numOwnedPoints,
      const double m_permeabilityScalar,
      const double m_permeabilityCurveInflectionDamage,
      const double m_permeabilityAlpha,
      const double m_maxPermeability,
      const double m_phaseOneBasePerm,
      const double m_phaseTwoBasePerm,
      const double m_phaseOneDensity,
      const double m_phaseTwoDensity,
      const double m_phaseOneCompressibility,
      const double m_phaseTwoCompressibility,
      const double m_phaseOneViscosity,
      const double m_phaseTwoViscosity,
      const double m_horizon,
      const double m_horizon_fracture,
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
      const double *principleDamageDirectionOwned = principleDamageDirection;
      const double *thetaCritical = criticalDilatationOwned;
      const std::complex<double> *thetaBreakless = breaklessDilatationOwned;

      std::complex<double> *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
      std::complex<double> *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;
      std::complex<double> *phaseTwoPoreFlowOwned = phaseTwoPoreFlowOverlap;
      std::complex<double> *phaseTwoFracFlowOwned = phaseTwoFracFlowOverlap;

      const int *neighPtr = localNeighborList;
      double cellVolume, harmonicAverageDamage;
      std::complex<double> phaseOnePorePerm, phaseTwoPorePerm;
      std::complex<double> dPorePressure, dFracPressure, dLocalPoreFracPressure, Y_dx, Y_dy, Y_dz, dY, fracPermeability;
      std::complex<double> fractureDirectionFactor, phaseOneFracPerm, phaseTwoFracPerm, phaseOneRelPermPores;
      std::complex<double> phaseOneRelPermFrac, phaseTwoRelPermPores, phaseTwoRelPermFrac, satStarPores, satStarFrac;
      std::complex<double> scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, scalarPhaseTwoPoreFlow, scalarPhaseTwoFracFlow;
      std::complex<double> scalarPhaseOneFracToPoreFlow, scalarPhaseTwoFracToPoreFlow, omegaPores, omegaFrac;

      for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                        phaseOneSaturationPoresYOwned++, phaseOneSaturationPoresVOwned++,
                                        phaseOneSaturationFracYOwned++, phaseOneSaturationFracVOwned++,
                                        yOwned +=3, phaseOnePoreFlowOwned++, phaseOneFracFlowOwned++,
                                        phaseTwoPoreFlowOwned++, phaseTwoFracFlowOwned++, deltaT++, damageOwned++,
                                        principleDamageDirectionOwned +=3, thetaCritical++, thetaBreakless++,
                                        porePressureVOwned++, fracturePressureVOwned++){
        int numNeigh = *neighPtr; neighPtr++;
        double selfCellVolume = v[p];
        const std::complex<double> *Y = yOwned;
        const std::complex<double> *porePressureY = porePressureYOwned;
        const std::complex<double> *porePressureV = porePressureVOwned;
        const std::complex<double> *fracturePressureY = fracturePressureYOwned;
        const std::complex<double> *fracturePressureV = fracturePressureVOwned;
        const std::complex<double> *phaseOneSaturationPoresY = phaseOneSaturationPoresYOwned;
        const std::complex<double> *phaseOneSaturationFracY = phaseOneSaturationFracYOwned;
        const double *principleDamageDirection = principleDamageDirectionOwned;

        // compute relative permeabilities assuming no damage effect
        satStarPores = (*phaseOneSaturationPoresY - 0.2)/0.6; // means spec one is water
        satStarFrac = (*phaseOneSaturationFracY - 0.2)/0.6;
        phaseOneRelPermPores = m_phaseOneBasePerm*pow(satStarPores, 2.0); //Empirical model, exponent is related to the material
        phaseOneRelPermFrac = m_phaseOneBasePerm*pow(satStarFrac, 2.0);
        phaseTwoRelPermPores = m_phaseTwoBasePerm*pow((-satStarPores+1.0),2.0);
        phaseTwoRelPermFrac = m_phaseTwoBasePerm*pow((-satStarFrac+1.0),2.0);

        // for to calculate Leakoff
        dLocalPoreFracPressure = *fracturePressureY - *porePressureY;

        //compute fracture width based on a two sphere diameter difference. An ad-hoc relation.
        if(std::real(*thetaBreakless) > 0.0){
          fracPermeability = pow(6.0/M_PI*selfCellVolume*(*thetaBreakless),1.0/3.0) - pow(6.0/M_PI*selfCellVolume*(*thetaCritical),1.0/3.0);
          if(std::real(fracPermeability) < 0.0)
            fracPermeability = std::complex<double>(0.0, std::imag(fracPermeability)); //Closed fractures have no real flow
        }
        else
          fracPermeability = std::complex<double>(0.0, std::imag(fracPermeability));

        fracPermeability *= fracPermeability/12.0; //Empirical relation for permeability

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
          omegaPores = exp(-dY*dY/(m_horizon*m_horizon));// scalarInfluenceFunction(dY,m_horizon);
          //Frac diffusion is a more local process than pore diffusion.
          omegaFrac = exp(-dY*dY/(m_horizon_fracture*m_horizon_fracture));// scalarInfluenceFunction(dY,m_horizon_fracture);

          // Pressure potential
          dPorePressure = *porePressureYP - *porePressureY;
          dFracPressure = *fracturePressureYP - *fracturePressureY;

          // compute permeabilities
          // Frac permeability in directions other than orthogonal to the principle damage direction is strongly attenuated.
          fractureDirectionFactor = pow(cos(Y_dx*(*(principleDamageDirection+0)) + Y_dy*(*(principleDamageDirection+1)) + Y_dz*(*(principleDamageDirection+2))),2.0); //Frac flow allowed in direction perpendicular to damage
          harmonicAverageDamage = 1.0 / (1.0 / *damageOwned + 1.0 / *damageNeighbor);
          if(harmonicAverageDamage != harmonicAverageDamage) harmonicAverageDamage=0.0; //test for nan which occurs when a damage is zero.
          // Pore permeability is affected by an ad-hoc S-curve relation to damage.
          phaseOnePorePerm = m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(exp(-m_permeabilityAlpha*(harmonicAverageDamage - m_permeabilityCurveInflectionDamage))+1.0);
          phaseTwoPorePerm = m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(exp(-m_permeabilityAlpha*(harmonicAverageDamage - m_permeabilityCurveInflectionDamage))+1.0);

          // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
          phaseOneFracPerm = fracPermeability*fractureDirectionFactor*phaseOneRelPermFrac;
          phaseTwoFracPerm = fracPermeability*fractureDirectionFactor*phaseTwoRelPermFrac;

          // compute flow density
          // flow entering cell is positive
          scalarPhaseOnePoreFlow = omegaPores * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseOnePorePerm / pow(dY, 4.0)) * dPorePressure;
          scalarPhaseOneFracFlow = omegaFrac * m_phaseOneDensity / m_phaseOneViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseOneFracPerm / pow(dY, 4.0)) * dFracPressure;
          scalarPhaseTwoPoreFlow = omegaPores * m_phaseTwoDensity / m_phaseTwoViscosity * (4.0 / (M_PI*m_horizon*m_horizon)) * (phaseTwoPorePerm / pow(dY, 4.0)) * dPorePressure;
          scalarPhaseTwoFracFlow = omegaFrac * m_phaseTwoDensity / m_phaseTwoViscosity * (4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture)) * (phaseTwoFracPerm / pow(dY, 4.0)) * dFracPressure;

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

        //Leakoff calculation, self to self
        *phaseOnePoreFlowOwned += m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
        *phaseOneFracFlowOwned -= m_phaseOneDensity / m_phaseOneViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseOneRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
        *phaseTwoPoreFlowOwned += m_phaseTwoDensity / m_phaseTwoViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;
        *phaseTwoFracFlowOwned -= m_phaseTwoDensity / m_phaseTwoViscosity * 4.0 / (M_PI*m_horizon_fracture*m_horizon_fracture) * (m_permeabilityScalar*phaseTwoRelPermPores + m_maxPermeability/(1.0 + exp(-m_permeabilityAlpha*(*damageOwned - m_permeabilityCurveInflectionDamage))) / pow(m_horizon_fracture, 4.0)) * dLocalPoreFracPressure;

        //Viscous terms
        double Porosity= 0.2;
        double B_formation_vol_factor_water = 1.0;
        double B_formation_vol_factor_oil = 1.0;
        double Compressibility_rock = 1.0;
        *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned - *porePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity*(*phaseOneSaturationPoresYOwned)/B_formation_vol_factor_water + Porosity/B_formation_vol_factor_water*(*phaseOneSaturationPoresVOwned);
        *phaseOneFracFlowOwned = *phaseOneFracFlowOwned - *fracturePressureV*m_phaseOneDensity*(Compressibility_rock + m_phaseOneCompressibility)*Porosity*(*phaseOneSaturationFracYOwned)/B_formation_vol_factor_water + Porosity/B_formation_vol_factor_water*(*phaseOneSaturationFracVOwned);
        *phaseTwoPoreFlowOwned = *phaseTwoPoreFlowOwned - *porePressureV*m_phaseTwoDensity*(Compressibility_rock + m_phaseTwoCompressibility)*Porosity*(1.0 - *phaseOneSaturationPoresYOwned)/B_formation_vol_factor_oil + Porosity/B_formation_vol_factor_oil*(*phaseOneSaturationPoresVOwned);
        *phaseTwoFracFlowOwned = *phaseTwoFracFlowOwned - *fracturePressureV*m_phaseTwoDensity*(Compressibility_rock + m_phaseTwoCompressibility)*Porosity*(1.0 - *phaseOneSaturationFracYOwned)/B_formation_vol_factor_oil + Porosity/B_formation_vol_factor_oil*(*phaseOneSaturationFracVOwned);
      }
    }

  }

}
