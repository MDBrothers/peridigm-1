//! \file one_phase_nonlocal_diffusion.cxx

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
#include "one_phase_nonlocal_diffusion.h"
#include "material_utilities.h"
#include <boost/math/special_functions/fpclassify.hpp>


namespace MATERIAL_EVALUATION {

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
  const ScalarT* matrixPorosityN,
  const ScalarT* fracturePorosityNP1,
  const ScalarT* fracturePorosityN,
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
  const ScalarT *matrixPorosityOwnedN = matrixPorosityN;
  const ScalarT *matrixPorosityOwnedNP1 = fracturePorosityNP1;
  const ScalarT *matrixPorosityOwnedN = fracturePorosityN;

  const ScalarT *thetaLocal = breaklessDilatation;
  ScalarT *phaseOnePoreFlowOwned = phaseOnePoreFlowOverlap;
  ScalarT *phaseOneFracFlowOwned = phaseOneFracFlowOverlap;

  const int *neighPtr = localNeighborList;
  double cellVolume, harmonicAverageDamage;
  ScalarT phaseOnePorePerm,  dPorePressure, dFracPressure;
  ScalarT dFracMinusPorePress, Y_dx, Y_dy, Y_dz, dY, fractureWidthFactor;
  ScalarT fractureDirectionFactor, phaseOneFracPerm, phaseOneRelPermPores,
  ScalarT scalarPhaseOnePoreFlow, scalarPhaseOneFracFlow, phaseOneRelPermFrac;
  ScalarT scalarPhaseOneFracToPoreFlow, omegaPores, omegaFrac;

  for(int p=0;p<numOwnedPoints;p++, porePressureYOwned++, fracturePressureYOwned++,
                                    yOwned +=3, phaseOnePoreFlowOwned++,
                                    phaseOneFracFlowOwned++, deltaT++,
                                    damageOwned++, thetaLocal++,
                                    principleDamageDirectionOwned +=3,
                                    matrixPorosityOwnedNP1++, fracturePorosityOwnedNP1++,
                                    matrixPorosityOwnedN++, fracturePorosityOwnedN++,
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
    fractureWidthFactor = (2.0*m_horizon*(*fracturePorosityOwnedNP1)*(2.0*m_horizon*(*fracturePorosityOwnedNP1)/12.0;

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
      fractureDirectionFactor = pow(cos(Y_dx*(*(principleDamageDirection+0)) + Y_dy*(*(principleDamageDirection+1)) + Y_dz*(*(principleDamageDirection+2))),2.0); //Frac flow allowed in direction perpendicular to damage
      // Frac permeability is affected by bond allignment with fracture plane, width, and saturation
      phaseOneFracPerm = fractureWidthFactor*fractureDirectionFactor;

      // compute flow density
      // flow entering cell is positive
      scalarPhaseOnePoreFlow = omegaPores * m_phaseOneDensity / m_phaseOneViscosity * m_permeabilityScalar / pow(dY, 4.0) * dPorePressure;
      scalarPhaseOneFracFlow = omegaFrac * m_phaseOneDensity / (2.0 * m_phaseOneViscosity) * phaseOneFracPerm / pow(dY, 2.0) * dFracPressure;

      // convert flow density to flow and account for reactions
      *phaseOnePoreFlowOwned += scalarPhaseOnePoreFlow*cellVolume;
      *phaseOneFracFlowOwned += scalarPhaseOneFracFlow*cellVolume;
      phaseOnePoreFlowOverlap[localId] -= scalarPhaseOnePoreFlow*selfCellVolume;
      phaseOneFracFlowOverlap[localId] -= scalarPhaseOneFracFlow*selfCellVolume;
    }

    //Add in viscous and leakoff terms from mass conservation equation
    *phaseOnePoreFlowOwned = *phaseOnePoreFlowOwned - selfCellVolume*m_phaseOneDensity*(*matrixPorosityNP1 - *matrixPorosityN)/deltaTime + selfCellVolume * m_permeabilityScalar * 4.0*M_PI*m_horizon_fracture*m_horizon_fracture * dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/2.0));
    *phaseOneFracFlowOwned = *phaseOneFracFlowOwned - selfCellVolume*m_phaseOneDensity*(*fracturePorosityNP1 - *fracturePorosityN)/deltaTime - selfCellVolume * m_permeabilityScalar * 4.0*M_PI*m_horizon_fracture*m_horizon_fracture * dFracMinusPorePress / (selfCellVolume*(1.0 + *thetaLocal)*m_phaseOneViscosity*(m_horizon/2.0));
  }
}

/** Explicit template instantiation for double. */
template void computeInternalFlow<double>
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
  const ScalarT* matrixPorosityN,
  const ScalarT* fracturePorosityNP1,
  const ScalarT* fracturePorosityN,
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
);

/** Explicit template instantiation for Sacado::Fad::DFad<double>. */
template void computeInternalFlow<Sacado::Fad::DFad<double> >
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
  const ScalarT* matrixPorosityN,
  const ScalarT* fracturePorosityNP1,
  const ScalarT* fracturePorosityN,
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
);

/** Explicit template instantiation for std::complex<double>. */
template void computeInternalFlow<std::complex<double> >
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
  const ScalarT* matrixPorosityN,
  const ScalarT* fracturePorosityNP1,
  const ScalarT* fracturePorosityN,
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
);

//! Compute the new matrix Porosity
template<typename ScalarT>
void computeMatrixPorosity
(
  ScalarT* matrixPorosityNP1,
  const ScalarT* matrixPorosityN,
  const ScalarT* porePressureYOverlapNP1,
  const ScalarT* porePressureYOverlapN,
  const ScalarT* dilatationNP1,
  const ScalarT* dilatationN,
  const double compressibilityRock,
  const double alpha,
  int numOwnedPoints
){
  ScalarT* phiMatrixNP1 = matrixPorosityNP1;
  const ScalarT* phiMatrixN = matrixPorosityN;
  const ScalarT* porePressureYOwnedNP1 = porePressureYOverlapNP1;
  const ScalarT* porePressureYOwnedN = porePressureYOverlapN;
  const ScalarT* thetaMatrixLocalNP1 = dilatationNP1; //The definition of matrix dilatation from Hisanao's formulation matches the standard definition of dilatation with a critical stretch damage model.
  const ScalarT* thetaMatrixLocalN = dilatationN;

  for(int p=0; p<numOwnedPoints; p++,  phiMatrixNP1++, phiMatrixN++, porePressureYOwnedNP1++, porePressureYOwnedN++, thetaMatrixLocalNP1++, thetaMatrixLocalN++){
    *phiMatrixNP1 = (*phiMatrixN)*(1.0 - compressibilityRock*(*porePressureYOwnedNP1 - *porePressureYOwnedN)) +
                    alpha*(1.0 + *thetaMatrixLocalN)*(compressibilityRock*(*porePressureYOwnedNP1 - *porePressureYOwnedN) + (*thetaMatrixLocalNP1 - *thetaMatrixLocalN));
  }
}

//! Explicit template instantiation for method to compute the new matrix Porosity
template void computeMatrixPorosity<std::complex<double> >
(
  std::complex<double>* matrixPorosityNP1,
  const std::complex<double>* matrixPorosityN,
  const std::complex<double>* porePressureYOverlapNP1,
  const std::complex<double>* porePressureYOverlapN,
  const std::complex<double>* dilatationNP1,
  const std::complex<double>* dilatationN,
  const double compressibilityRock,
  const double alpha,
  int numOwnedPoints
);

//! Explicit template intstantiation for method to compute the new matrix Porosity
template void computeMatrixPorosity<double>
(
  double* matrixPorosityNP1,
  const double* matrixPorosityN,
  const double* porePressureYOverlapNP1,
  const double* porePressureYOverlapN,
  const double* dilatationNP1,
  const double* dilatationN,
  const double compressibilityRock,
  const double alpha,
  const int* localNeighborList,
  int numOwnedPoints
);

template<typename ScalarT>
void computeFracturePorosity
(
  ScalarT* fracturePorosityNP1,
  const ScalarT* breaklessDilatationOwnedNP1,
  const double* criticalDilatationOwned,
  const int* localNeighborList,
  int numOwnedPoints
){
  const ScalarT* thetaLocal = breaklessDilatationOwnedNP1; //The definition of local dilatation from Hisanao's formulation matches the standard definition of dilatation without a damage model.
  const double* thetaCritical = criticalDilatationOwned;
  ScalarT* fracturePorosity = fracturePorosityNP1;

  for(int p=0; p<numOwnedPoints;p++, thetaLocal++, thetaCritical++, fracturePorosity++){
    *fracturePorosity = *thetaLocal - *thetaCritical;
    if(*fracturePorosity < 0.0) *fracturePorosity = 0.0; //No negative porosities.
  }
}

//! Explicit template specialization for to compute the new fracture Porosity
template void computeFracturePorosity<std::complex<double> >
(
  std::complex<double>* fracturePorosityNP1,
  const std::complex<double>* breaklessDilatationOwnedNP1,
  const double* criticalDilatationOwned,
  int numOwnedPoints
){
  const std::complex<double>* thetaLocal = breaklessDilatationOwnedNP1; //The definition of local dilatation from Hisanao's formulation matches the standard definition of dilatation without a damage model.
  const double* thetaCritical = criticalDilatationOwned;
  std::complex<double>* fracturePorosity = fracturePorosityNP1;

  for(int p=0; p<numOwnedPoints;p++, thetaLocal++, thetaCritical++, fracturePorosity++){
    *fracturePorosity = *thetaLocal - *thetaCritical;
    if(std::real(*fracturePorosity) < 0.0) *fracturePorosity = std::complex(0.0, std::imag(*fracturePorosity)); //No negative porosities.
  }
}

//! Explcit template instantiation for to compute the fracture Porosity
template void computeFracturePorosity<double>
(
  double* fracturePorosityNP1,
  const double* breaklessDilatationOwnedNP1,
  const double* criticalDilatationOwned,
  int numOwnedPoints
);

template<typename ScalarT>
void computePhaseOneDensityInPores
(
  ScalarT* phaseOneDensityInPores,
  const ScalarT* porePressureYOVerlap,
  const double* deltaTemperature,
  int numOwnedPoints
)
//Please see the following code. (pressure [Pa], Temperature [K], density: water density [kg/m3], enthalpy: water enthalpy [J/kg], internalEnergy: water internal energy[J/kg])
inline void get_water_prop_2comp_simple_peri(double Pressure, double Temperature,  double &density, double &enthalpy, double &internalEnergy)
{
	double P;
	double dtemp, dtemp2;

	P = Pressure*1.0e-6;
	dtemp = P*P;
	dtemp2 = Temperature*Temperature;

	density = (-0.00000014569010515*dtemp + 0.000046724532297*P - 0.0061488874609)*dtemp2
		+ (0.000088493144499*dtemp - 0.029002566308*P + 3.3982146161)*Temperature
		- 0.013875092279*dtemp + 4.9439957018*P + 530.4110022;
	//density = 1000*(1+5e-10*(Pressure-0.1e+6));
	//density = 1000;
	enthalpy = (-0.00021992452509*dtemp + 0.037121520627*P - 0.62144893236)*dtemp2 + (0.14041054093*dtemp - 24.86452127*P + 4549.1148295)*Temperature - 22.685476185*dtemp + 5039.7750745*P - 1196243.3551;
	internalEnergy = enthalpy - Pressure / density;
	//internalEnergy = enthalpy;
}

template<typename ScalarT>
void computePhaseOneDensityInFracture
(
  ScalarT* phaseOneDensityInFracture,
  const ScalarT* fracturePRessureYOverlap,
  const double* deltaTemperature
);
