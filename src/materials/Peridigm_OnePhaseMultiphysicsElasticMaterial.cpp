/*! \file Peridigm_OnePhaseMultiphysicsElasticMaterial.cpp */

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
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, PhaseIAL,
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

#include "Peridigm_OnePhaseMultiphysicsElasticMaterial.hpp"
#include "Peridigm_Field.hpp"

#include "poroelastic.hpp"
#include "diffusion_models.hpp"
#include "damage_model_dependent_utilities.hpp"
#include "material_utilities.h"

#include <Teuchos_Assert.hpp>
#include <Epetra_SerialComm.h>
#include <Sacado.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

using namespace std;

PeridigmNS::OnePhaseMultiphysicsElasticMaterial::OnePhaseMultiphysicsElasticMaterial(const Teuchos::ParameterList& params)
  : Material(params),
    m_bulkModulus(0.0),
    m_shearModulus(0.0),
    m_density(0.0),
    m_alpha(0.0),
    m_horizon(0.0),
    m_horizon_fracture(0.0),
    m_matrixPermeabilityXX(0.0),
    m_matrixPermeabilityYY(0.0),
    m_matrixPermeabilityZZ(0.0),
    m_phaseOneViscosity(0.0),
    m_criticalStretch(0.0),
    m_alphaBiot(0.0),
    m_compressibilityRock(0.0),
    m_applyAutomaticDifferentiationJacobian(false),
    m_applySurfaceCorrectionFactor(false),
    m_applyThermalStrains(false),
    m_OMEGA(PeridigmNS::InfluenceFunction::self().getInfluenceFunction()),
    m_volumeFieldId(-1),
    m_damageFieldId(-1),
    m_weightedVolumeFieldId(-1),
    m_dilatationFieldId(-1),
    m_modelCoordinatesFieldId(-1),
    m_coordinatesFieldId(-1),
    m_forceDensityFieldId(-1),
    m_bondDamageFieldId(-1),
    m_surfaceCorrectionFactorFieldId(-1),
    m_deltaTemperatureFieldId(-1),
    m_matrixPorosityFieldId(-1),
    m_fracturePorosityFieldId(-1),
    m_phaseOneDensityInPoresFieldId(-1),
    m_phaseOneDensityInFractureFieldId(-1),
    m_porePressureYFieldId(-1),
    m_porePressureVFieldId(-1),
    m_fracturePressureYFieldId(-1),
    m_fracturePressureVFieldId(-1),
    m_phaseOnePoreFlowDensityFieldId(-1),
    m_phaseOneFracFlowDensityFieldId(-1)

{
  //! \todo Add meaningful asserts on material properties.
  m_bulkModulus = calculateBulkModulus(params);
  m_shearModulus = calculateShearModulus(params);
  m_density = params.get<double>("Density");
  m_horizon = params.get<double>("Horizon");
  m_horizon_fracture = params.get<double>("Frac Diffusion Horizon");
	m_matrixPermeabilityXX = params.get<double>("Matrix Permeability XX");
  m_matrixPermeabilityYY = params.get<double>("Matrix Permeability YY");
  m_matrixPermeabilityZZ = params.get<double>("Matrix Permeability ZZ");
	m_phaseOneViscosity = params.get<double>("Phase One Viscosity");
	m_criticalStretch = params.get<double>("Material Duplicate Critical Stretch");
  m_alphaBiot = params.get<double>("Poroelasticity Constant");
  m_compressibilityRock = params.get<double>("Compressibility Rock");

  if(params.isParameter("Apply Automatic Differentiation Jacobian"))
    m_applyAutomaticDifferentiationJacobian = params.get<bool>("Apply Automatic Differentiation Jacobian");

  if(params.isParameter("Apply Shear Correction Factor"))
    m_applySurfaceCorrectionFactor = params.get<bool>("Apply Shear Correction Factor");

  if(params.isParameter("Thermal Expansion Coefficient")){
    m_alpha = params.get<double>("Thermal Expansion Coefficient");
  materialProperties["Solid thermal expansion coefficient"] = m_alpha;
    m_applyThermalStrains = true;
  }

  TEUCHOS_TEST_FOR_EXCEPT_MSG((m_applyAutomaticDifferentiationJacobian ), "**** Error: One Phase Multiphysics Elastic currently supports only a complex step method jacobian.\n");

  PeridigmNS::FieldManager& fieldManager = PeridigmNS::FieldManager::self();
  m_volumeFieldId                  = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::CONSTANT, "Volume");
  m_damageFieldId                  = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Damage");
  m_weightedVolumeFieldId          = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::CONSTANT, "Weighted_Volume");
  m_dilatationFieldId              = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Dilatation");
  m_modelCoordinatesFieldId        = fieldManager.getFieldId(PeridigmField::NODE,    PeridigmField::VECTOR, PeridigmField::CONSTANT, "Model_Coordinates");
  m_coordinatesFieldId             = fieldManager.getFieldId(PeridigmField::NODE,    PeridigmField::VECTOR, PeridigmField::TWO_STEP, "Coordinates");
  m_forceDensityFieldId            = fieldManager.getFieldId(PeridigmField::NODE,    PeridigmField::VECTOR, PeridigmField::TWO_STEP, "Force_Density");
  m_bondDamageFieldId              = fieldManager.getFieldId(PeridigmField::BOND,    PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Bond_Damage");
  m_surfaceCorrectionFactorFieldId = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::CONSTANT, "Surface_Correction_Factor");

  //Multiphysics field variables
  m_criticalDilatationFieldId = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::CONSTANT, "Critical_Dilatation");
  m_breaklessDiltatationFieldId = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Breakless_Dilatation");
  m_porePressureYFieldId = fieldManager.getFieldId(PeridigmField::NODE, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Pore_Pressure_Y");
  m_porePressureVFieldId = fieldManager.getFieldId(PeridigmField::NODE, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Pore_Pressure_V");
  m_fracturePressureYFieldId = fieldManager.getFieldId(PeridigmField::NODE, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Frac_Pressure_Y");
  m_fracturePressureVFieldId = fieldManager.getFieldId(PeridigmField::NODE, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Frac_Pressure_V");
  m_phaseOnePoreFlowDensityFieldId = fieldManager.getFieldId(PeridigmField::NODE, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Phase_1_Pore_Flow_Density");
  m_phaseOneFracFlowDensityFieldId  = fieldManager.getFieldId(PeridigmField::NODE, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Phase_1_Frac_Flow_Density");
  m_matrixPorosityFieldId = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Matrix_Porosity");
  m_fracturePorosityFieldId = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Fracture_Porosity");
  m_phaseOneDensityInPoresFieldId = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Phase_1_Density_In_Pores");
  m_phaseOneDensityInFractureFieldId = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Phase_1_Density_In_Fracture");

  if(m_applyThermalStrains)
    m_deltaTemperatureFieldId = fieldManager.getFieldId(PeridigmField::NODE, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Temperature_Change");

  m_fieldIds.push_back(m_volumeFieldId);
  m_fieldIds.push_back(m_damageFieldId);
  m_fieldIds.push_back(m_weightedVolumeFieldId);
  m_fieldIds.push_back(m_dilatationFieldId);
  m_fieldIds.push_back(m_criticalDilatationFieldId);
  m_fieldIds.push_back(m_breaklessDiltatationFieldId);
  m_fieldIds.push_back(m_criticalDilatationFieldId);
  m_fieldIds.push_back(m_breaklessDiltatationFieldId);
  m_fieldIds.push_back(m_porePressureYFieldId);
  m_fieldIds.push_back(m_porePressureVFieldId);
  m_fieldIds.push_back(m_fracturePressureYFieldId);
  m_fieldIds.push_back(m_fracturePressureVFieldId);
  m_fieldIds.push_back(m_matrixPorosityFieldId);
  m_fieldIds.push_back(m_fracturePorosityFieldId);
  m_fieldIds.push_back(m_phaseOneDensityInPoresFieldId);
  m_fieldIds.push_back(m_phaseOneDensityInFractureFieldId);
  m_fieldIds.push_back(m_phaseOnePoreFlowDensityFieldId);
  m_fieldIds.push_back(m_phaseOneFracFlowDensityFieldId);
  m_fieldIds.push_back(m_modelCoordinatesFieldId);
  m_fieldIds.push_back(m_coordinatesFieldId);
  m_fieldIds.push_back(m_forceDensityFieldId);
  m_fieldIds.push_back(m_bondDamageFieldId);
  m_fieldIds.push_back(m_surfaceCorrectionFactorFieldId);
  if(m_applyThermalStrains)
    m_fieldIds.push_back(m_deltaTemperatureFieldId);
}

PeridigmNS::OnePhaseMultiphysicsElasticMaterial::~OnePhaseMultiphysicsElasticMaterial()
{
}

double
PeridigmNS::OnePhaseMultiphysicsElasticMaterial::lookupMaterialProperty(const std::string keyname) const
{
  std::map<std::string, double>::const_iterator search = materialProperties.find(keyname);
  if(search != materialProperties.end())
    return search->second;
  else
    TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "**** Error: requested material property is not in Multiphysics Elastic Material");
  // This is a fallthrough case to make the compiler happy.
  return 0.0;
}

void
PeridigmNS::OnePhaseMultiphysicsElasticMaterial::initialize(const double dt,
                                        const int numOwnedPoints,
                                        const int* ownedIDs,
                                        const int* neighborhoodList,
                                        PeridigmNS::DataManager& dataManager)
{
  // Extract pointers to the underlying data
  double *xOverlap,  *cellVolumeOverlap, *weightedVolume, *criticalDilatation;
  dataManager.getData(m_modelCoordinatesFieldId, PeridigmField::STEP_NONE)->ExtractView(&xOverlap);
  dataManager.getData(m_volumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&cellVolumeOverlap);
  dataManager.getData(m_weightedVolumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&weightedVolume);
  dataManager.getData(m_criticalDilatationFieldId, PeridigmField::STEP_NONE)->ExtractView(&criticalDilatation);

  //Trick the the computer into allocating memory for these fields, inspired by the correspondence model.
  dataManager.getData(m_criticalDilatationFieldId, PeridigmField::STEP_NONE)->PutScalar(0.0);
  dataManager.getData(m_breaklessDiltatationFieldId, PeridigmField::STEP_NP1)->PutScalar(0.0);
  dataManager.getData(m_breaklessDiltatationFieldId, PeridigmField::STEP_N)->PutScalar(0.0);

  MATERIAL_EVALUATION::computeWeightedVolume(xOverlap,cellVolumeOverlap,weightedVolume,numOwnedPoints,neighborhoodList,m_horizon);
  MATERIAL_EVALUATION::computeCriticalDilatation(xOverlap,weightedVolume,cellVolumeOverlap,criticalDilatation,neighborhoodList,numOwnedPoints,m_horizon,m_criticalStretch,m_OMEGA);
}

void
PeridigmNS::OnePhaseMultiphysicsElasticMaterial::computeForce(const double dt,
                                          const int numOwnedPoints,
                                          const int* ownedIDs,
                                          const int* neighborhoodList,
                                          PeridigmNS::DataManager& dataManager) const
{
  // Zero out the forces
  dataManager.getData(m_forceDensityFieldId, PeridigmField::STEP_NP1)->PutScalar(0.0);
  dataManager.getData(m_phaseOnePoreFlowDensityFieldId, PeridigmField::STEP_NP1)->PutScalar(0.0);
  dataManager.getData(m_phaseOneFracFlowDensityFieldId, PeridigmField::STEP_NP1)->PutScalar(0.0);

  // Extract pointers to the underlying data
  double *x, *y, *cellVolume, *weightedVolume, *dilatation, *criticalDilatation, *breaklessDilatation;
  double *bondDamage, *force, *scf, *deltaTemperature;
  double *damage, *porePressureY, *porePressureV, *fracturePressureY, *fracturePressureV;
  double *phaseOnePoreFlow, *phaseOneFracFlow;
  double *phaseOneDensityInPoresN, *phaseOneDensityInPoresNP1, *phaseOneDensityInFractureN, *phaseOneDensityInFractureNP1;
  double *matrixPorosityN, *matrixPorosityNP1, *dilatationN, *porePressureYN, *fracturePressureYN;
  double *fracturePorosityNP1, *fracturePorosityN;

  dataManager.getData(m_modelCoordinatesFieldId, PeridigmField::STEP_NONE)->ExtractView(&x);
  dataManager.getData(m_coordinatesFieldId, PeridigmField::STEP_NP1)->ExtractView(&y);
  dataManager.getData(m_volumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&cellVolume);
  dataManager.getData(m_weightedVolumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&weightedVolume);
  dataManager.getData(m_dilatationFieldId, PeridigmField::STEP_NP1)->ExtractView(&dilatation);
  dataManager.getData(m_dilatationFieldId, PeridigmField::STEP_N)->ExtractView(&dilatationN);
  dataManager.getData(m_criticalDilatationFieldId, PeridigmField::STEP_NONE)->ExtractView(&criticalDilatation);
  dataManager.getData(m_breaklessDiltatationFieldId, PeridigmField::STEP_NP1)->ExtractView(&breaklessDilatation);
  dataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_NP1)->ExtractView(&bondDamage);
  dataManager.getData(m_damageFieldId, PeridigmField::STEP_NP1)->ExtractView(&damage);
  dataManager.getData(m_forceDensityFieldId, PeridigmField::STEP_NP1)->ExtractView(&force);
  dataManager.getData(m_surfaceCorrectionFactorFieldId, PeridigmField::STEP_NONE)->ExtractView(&scf);
  dataManager.getData(m_porePressureYFieldId, PeridigmField::STEP_NP1)->ExtractView(&porePressureY);
  dataManager.getData(m_porePressureYFieldId, PeridigmField::STEP_N)->ExtractView(&porePressureYN);
  dataManager.getData(m_porePressureVFieldId, PeridigmField::STEP_NP1)->ExtractView(&porePressureV);
  dataManager.getData(m_fracturePressureYFieldId, PeridigmField::STEP_NP1)->ExtractView(&fracturePressureY);
  dataManager.getData(m_fracturePressureYFieldId, PeridigmField::STEP_N)->ExtractView(&fracturePressureYN);
  dataManager.getData(m_fracturePressureVFieldId, PeridigmField::STEP_NP1)->ExtractView(&fracturePressureV);
  dataManager.getData(m_phaseOnePoreFlowDensityFieldId, PeridigmField::STEP_NP1)->ExtractView(&phaseOnePoreFlow);
  dataManager.getData(m_phaseOneFracFlowDensityFieldId, PeridigmField::STEP_NP1)->ExtractView(&phaseOneFracFlow);
  dataManager.getData(m_phaseOneDensityInPoresFieldId, PeridigmField::STEP_N)->ExtractView(&phaseOneDensityInPoresN);
  dataManager.getData(m_phaseOneDensityInPoresFieldId, PeridigmField::STEP_NP1)->ExtractView(&phaseOneDensityInPoresNP1);
  dataManager.getData(m_phaseOneDensityInFractureFieldId, PeridigmField::STEP_N)->ExtractView(&phaseOneDensityInFractureN);
  dataManager.getData(m_phaseOneDensityInFractureFieldId, PeridigmField::STEP_NP1)->ExtractView(&phaseOneDensityInFractureNP1);

  dataManager.getData(m_matrixPorosityFieldId, PeridigmField::STEP_NP1)->ExtractView(&matrixPorosityNP1);
  dataManager.getData(m_matrixPorosityFieldId, PeridigmField::STEP_N)->ExtractView(&matrixPorosityN);
  dataManager.getData(m_fracturePorosityFieldId, PeridigmField::STEP_NP1)->ExtractView(&fracturePorosityNP1);
  dataManager.getData(m_fracturePorosityFieldId, PeridigmField::STEP_N)->ExtractView(&fracturePorosityN);

  deltaTemperature = NULL;
  if(m_applyThermalStrains) dataManager.getData(m_deltaTemperatureFieldId, PeridigmField::STEP_NP1)->ExtractView(&deltaTemperature);

  MATERIAL_EVALUATION::computeBreaklessDilatation(x,y,weightedVolume,cellVolume,breaklessDilatation,neighborhoodList,numOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
  MATERIAL_EVALUATION::computeDilatation(x,y,weightedVolume,cellVolume,bondDamage,dilatation,neighborhoodList,numOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
  MATERIAL_EVALUATION::computeMatrixPorosity(matrixPorosityNP1,matrixPorosityN,porePressureY,porePressureYN,dilatation,dilatationN,m_compressibilityRock,m_alphaBiot,numOwnedPoints);
  MATERIAL_EVALUATION::computeFracturePorosity(fracturePorosityNP1,breaklessDilatation,criticalDilatation,numOwnedPoints);
  MATERIAL_EVALUATION::computePhaseOneDensityInPores(phaseOneDensityInPoresNP1,porePressureY,deltaTemperature,numOwnedPoints);
  MATERIAL_EVALUATION::computePhaseOneDensityInFracture(phaseOneDensityInFractureNP1,fracturePressureY,deltaTemperature,numOwnedPoints);
  MATERIAL_EVALUATION::computeInternalForceLinearElasticCoupled(x,y,porePressureY,fracturePressureY,weightedVolume,cellVolume,dilatation,damage,bondDamage,scf,force,neighborhoodList,numOwnedPoints,m_bulkModulus,m_shearModulus,m_horizon,m_alphaBiot,m_alpha,deltaTemperature);
  MATERIAL_EVALUATION::ONE_PHASE::computeInternalFlow(y,porePressureY,porePressureV,fracturePressureY,fracturePressureV,cellVolume,damage,matrixPorosityNP1,matrixPorosityN,fracturePorosityNP1,fracturePorosityN,phaseOneDensityInPoresNP1,phaseOneDensityInPoresN,phaseOneDensityInFractureNP1,phaseOneDensityInFractureN,breaklessDilatation,phaseOnePoreFlow,phaseOneFracFlow,neighborhoodList,numOwnedPoints,m_matrixPermeabilityXX,m_matrixPermeabilityYY,m_matrixPermeabilityZZ,m_phaseOneViscosity,m_horizon,m_horizon_fracture,dt,deltaTemperature);
}

void
PeridigmNS::OnePhaseMultiphysicsElasticMaterial::computeStoredElasticEnergyDensity(const double dt,
                                                               const int numOwnedPoints,
                                                               const int* ownedIDs,
                                                               const int* neighborhoodList,
                                                               PeridigmNS::DataManager& dataManager) const
{
  // This function is intended to be called from a compute class.
  // The compute class should have already created the Stored_Elastic_Energy_Density field id.
  int storedElasticEnergyDensityFieldId = PeridigmNS::FieldManager::self().getFieldId("Stored_Elastic_Energy_Density");

  double *x, *y, *cellVolume, *weightedVolume, *dilatation, *storedElasticEnergyDensity, *bondDamage, *surfaceCorrectionFactor;
  dataManager.getData(m_modelCoordinatesFieldId, PeridigmField::STEP_NONE)->ExtractView(&x);
  dataManager.getData(m_coordinatesFieldId, PeridigmField::STEP_NP1)->ExtractView(&y);
  dataManager.getData(m_volumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&cellVolume);
  dataManager.getData(m_weightedVolumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&weightedVolume);
  dataManager.getData(m_dilatationFieldId, PeridigmField::STEP_NP1)->ExtractView(&dilatation);
  dataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_NP1)->ExtractView(&bondDamage);
  dataManager.getData(storedElasticEnergyDensityFieldId, PeridigmField::STEP_NONE)->ExtractView(&storedElasticEnergyDensity);
  dataManager.getData(m_surfaceCorrectionFactorFieldId, PeridigmField::STEP_NONE)->ExtractView(&surfaceCorrectionFactor);

  double *deltaTemperature = NULL;
  if(m_applyThermalStrains)
    dataManager.getData(m_deltaTemperatureFieldId, PeridigmField::STEP_NP1)->ExtractView(&deltaTemperature);

  int iID, iNID, numNeighbors, nodeId, neighborId;
  double omega, nodeInitialX[3], nodeCurrentX[3];
  double initialDistance, currentDistance, deviatoricExtension, neighborBondDamage;
  double nodeDilatation, alpha, temp;

  int neighborhoodListIndex(0), bondIndex(0);
  for(iID=0 ; iID<numOwnedPoints ; ++iID){

    nodeId = ownedIDs[iID];
    nodeInitialX[0] = x[nodeId*3];
    nodeInitialX[1] = x[nodeId*3+1];
    nodeInitialX[2] = x[nodeId*3+2];
    nodeCurrentX[0] = y[nodeId*3];
    nodeCurrentX[1] = y[nodeId*3+1];
    nodeCurrentX[2] = y[nodeId*3+2];
    nodeDilatation = dilatation[nodeId];
    alpha = 15.0*m_shearModulus/weightedVolume[nodeId];
    alpha *= surfaceCorrectionFactor[nodeId];

    temp = 0.0;

    numNeighbors = neighborhoodList[neighborhoodListIndex++];
    for(iNID=0 ; iNID<numNeighbors ; ++iNID){
      neighborId = neighborhoodList[neighborhoodListIndex++];
      neighborBondDamage = bondDamage[bondIndex++];
      initialDistance =
        distance(nodeInitialX[0], nodeInitialX[1], nodeInitialX[2],
                 x[neighborId*3], x[neighborId*3+1], x[neighborId*3+2]);
      currentDistance =
        distance(nodeCurrentX[0], nodeCurrentX[1], nodeCurrentX[2],
                 y[neighborId*3], y[neighborId*3+1], y[neighborId*3+2]);
      if(m_applyThermalStrains)
  currentDistance -= m_alpha*deltaTemperature[nodeId]*initialDistance;
      deviatoricExtension = (currentDistance - initialDistance) - nodeDilatation*initialDistance/3.0;
      omega=m_OMEGA(initialDistance,m_horizon);
      temp += (1.0-neighborBondDamage)*omega*deviatoricExtension*deviatoricExtension*cellVolume[neighborId];
    }
    storedElasticEnergyDensity[nodeId] = 0.5*m_bulkModulus*nodeDilatation*nodeDilatation + 0.5*alpha*temp;
  }
}

void
PeridigmNS::OnePhaseMultiphysicsElasticMaterial::computeJacobian(const double dt,
                                             const int numOwnedPoints,
                                             const int* ownedIDs,
                                             const int* neighborhoodList,
                                             PeridigmNS::DataManager& dataManager,
                                             PeridigmNS::SerialMatrix& jacobian,
                                             PeridigmNS::Material::JacobianType jacobianType) const
{
  //computeAutomaticDifferentiationJacobian(dt, numOwnedPoints, ownedIDs, neighborhoodList, dataManager, jacobian, jacobianType);
  //NOTE Phase two pore or fracture flow has a relationship with saturation in the fracture component only when the d/dt of pore or fracture pressure or d/dt of saturation are nonzero.
  //Under common loading conditions (zero pressure gradient), these requirements are violated which causes Automatic Differentiation to
  //"correctly" return zero for d/d(pore or fracture sat) of phase 2 fracture flow on the diagonal. Only a probing method like finite difference produces a nonsingular
  // Jacobian.
  computeComplexStepFiniteDifferenceJacobian(dt, numOwnedPoints, ownedIDs, neighborhoodList, dataManager, jacobian, jacobianType);
}

void
PeridigmNS::OnePhaseMultiphysicsElasticMaterial::computeAutomaticDifferentiationJacobian(const double dt,
                                                                     const int numOwnedPoints,
                                                                     const int* ownedIDs,
                                                                     const int* neighborhoodList,
                                                                     PeridigmNS::DataManager& dataManager,
                                                                     PeridigmNS::SerialMatrix& jacobian,
                                                                     PeridigmNS::Material::JacobianType jacobianType) const
{
  //NOTE this commented out code represents a pattern for implementation, but is not an up to date implementation.
  //This code will be deleted in a future commit and stay available in the commit history.
  // // To reduce memory re-allocation, use static variable to store Fad types for
  // // current coordinates (independent variables).
  // static vector<Sacado::Fad::DFad<double> > y_AD;
  // static vector<Sacado::Fad::DFad<double> > porePressureY_AD;
  // static vector<Sacado::Fad::DFad<double> > fracturePressureY_AD;
  //
  // // Loop over all points.
  // int neighborhoodListIndex = 0;
  // for(int iID=0 ; iID<numOwnedPoints ; ++iID){
  //
  //   // Create a temporary neighborhood consisting of a single point and its neighbors.
  //   int numNeighbors = neighborhoodList[neighborhoodListIndex++];
  //   int numEntries = numNeighbors+1;
  //   int numTotalNeighborhoodDof = 5*numEntries;
  //   vector<int> tempMyGlobalIDs(numEntries);
  //   // Put the node at the center of the neighborhood at the beginning of the list.
  //   tempMyGlobalIDs[0] = dataManager.getOwnedScalarPointMap()->GID(iID);
  //   vector<int> tempNeighborhoodList(numEntries);
  //   tempNeighborhoodList[0] = numNeighbors;
  //   for(int iNID=0 ; iNID<numNeighbors ; ++iNID){
  //     int neighborID = neighborhoodList[neighborhoodListIndex++];
  //     tempMyGlobalIDs[iNID+1] = dataManager.getOverlapScalarPointMap()->GID(neighborID);
  //     tempNeighborhoodList[iNID+1] = iNID+1;
  //   }
  //
  //   Epetra_SerialComm serialComm;
  //   Teuchos::RCP<Epetra_BlockMap> tempOneDimensionalMap = Teuchos::rcp(new Epetra_BlockMap(numEntries, numEntries, &tempMyGlobalIDs[0], 1, 0, serialComm));
  //   Teuchos::RCP<Epetra_BlockMap> tempThreeDimensionalMap = Teuchos::rcp(new Epetra_BlockMap(numEntries, numEntries, &tempMyGlobalIDs[0], 3, 0, serialComm));
  //   Teuchos::RCP<Epetra_BlockMap> tempBondMap = Teuchos::rcp(new Epetra_BlockMap(1, 1, &tempMyGlobalIDs[0], numNeighbors, 0, serialComm));
  //
  //   // Create a temporary DataManager containing data for this point and its neighborhood.
  //   PeridigmNS::DataManager tempDataManager;
  //   tempDataManager.setMaps(Teuchos::RCP<const Epetra_BlockMap>(),
  //                           tempOneDimensionalMap,
  //                           Teuchos::RCP<const Epetra_BlockMap>(),
  //                           tempThreeDimensionalMap,
  //                           tempBondMap);
  //
  //   // The temporary data manager will have the same field specs and data as the real data manager.
  //   vector<int> fieldIds = dataManager.getFieldIds();
  //   tempDataManager.allocateData(fieldIds);
  //   tempDataManager.copyLocallyOwnedDataFromDataManager(dataManager);
  //
  //   // Set up numOwnedPoints and ownedIDs.
  //   // There is only one owned ID, and it has local ID zero in the tempDataManager.
  //   int tempNumOwnedPoints = 1;
  //   vector<int> tempOwnedIDs(tempNumOwnedPoints);
  //   tempOwnedIDs[0] = 0;
  //
  //       // Use the scratchMatrix as sub-matrix for storing tangent values prior to loading them into the global tangent matrix.
  //   // Resize scratchMatrix if necessary
  //   if(scratchMatrix.Dimension() < numTotalNeighborhoodDof)
  //     scratchMatrix.Resize(numTotalNeighborhoodDof);
  //
  //   // Create a list of global indices for the rows/columns in the scratch matrix.
  //   vector<int> globalIndices(numTotalNeighborhoodDof);
  //   for(int i=0 ; i<numEntries ; ++i){
  //     int globalID = tempOneDimensionalMap->GID(i);
  //     for(int j=0 ; j<5 ; ++j)
  //       globalIndices[5*i+j] = 5*globalID+j;
  //   }
  //
  //   // Extract pointers to the underlying data
  //   double *x, *y, *cellVolume, *weightedVolume, *dilatation, *criticalDilatation, *breaklessDilatation;
  //   double *bondDamage, *force, *scf, *deltaTemperature;
  //   double *damage, *porePressureY, *porePressureV, *fracturePressureY, *fracturePressureV;
  //   double *phaseOnePoreFlow, *phaseOneFracFlow;
  //
  //   tempDataManager.getData(m_modelCoordinatesFieldId, PeridigmField::STEP_NONE)->ExtractView(&x);
  //   tempDataManager.getData(m_coordinatesFieldId, PeridigmField::STEP_NP1)->ExtractView(&y);
  //   tempDataManager.getData(m_volumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&cellVolume);
  //   tempDataManager.getData(m_weightedVolumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&weightedVolume);
  //   tempDataManager.getData(m_dilatationFieldId, PeridigmField::STEP_NP1)->ExtractView(&dilatation);
  //   tempDataManager.getData(m_criticalDilatationFieldId, PeridigmField::STEP_NONE)->ExtractView(&criticalDilatation);
  //   tempDataManager.getData(m_breaklessDiltatationFieldId, PeridigmField::STEP_NP1)->ExtractView(&breaklessDilatation);
  //   tempDataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_NP1)->ExtractView(&bondDamage);
  //   tempDataManager.getData(m_damageFieldId, PeridigmField::STEP_NP1)->ExtractView(&damage);
  //   tempDataManager.getData(m_forceDensityFieldId, PeridigmField::STEP_NP1)->ExtractView(&force);
  //   tempDataManager.getData(m_surfaceCorrectionFactorFieldId, PeridigmField::STEP_NONE)->ExtractView(&scf);
  //   tempDataManager.getData(m_porePressureYFieldId, PeridigmField::STEP_NP1)->ExtractView(&porePressureY);
  //   tempDataManager.getData(m_porePressureVFieldId, PeridigmField::STEP_NP1)->ExtractView(&porePressureV);
  //   tempDataManager.getData(m_fracturePressureYFieldId, PeridigmField::STEP_NP1)->ExtractView(&fracturePressureY);
  //   tempDataManager.getData(m_fracturePressureVFieldId, PeridigmField::STEP_NP1)->ExtractView(&fracturePressureV);
  //   tempDataManager.getData(m_phaseOnePoreFlowDensityFieldId, PeridigmField::STEP_NP1)->ExtractView(&phaseOnePoreFlow);
  //   tempDataManager.getData(m_phaseOneFracFlowDensityFieldId, PeridigmField::STEP_NP1)->ExtractView(&phaseOneFracFlow);
  //
  //   deltaTemperature = NULL;
  //   if(m_applyThermalStrains)
  //     tempDataManager.getData(m_deltaTemperatureFieldId, PeridigmField::STEP_NP1)->ExtractView(&deltaTemperature);
  //   // Create arrays of Fad objects for the current coordinates, dilatation, and force density
  //   // Modify the existing vector of Fad objects for the current coordinates
  //   if((int)y_AD.size() < (3*numEntries)){
  //     y_AD.resize(3*numEntries);
  //     porePressureY_AD.resize(numEntries);
  //     fracturePressureY_AD.resize(numEntries);
  //   }
  //
  //   for(int i=0 ; i< numTotalNeighborhoodDof ; i+=5){
  //     for(int j=0 ; j<3 ; ++j){
  //       y_AD[i*3/5+j].diff(i+j, numTotalNeighborhoodDof);
  //       y_AD[i*3/5+j].val() = y[i*3/5+j];
  //     }
  //     porePressureY_AD[i/5].diff(i+3,numTotalNeighborhoodDof);
  //     fracturePressureY_AD[i/5].diff(i+4,numTotalNeighborhoodDof);
  //     porePressureY_AD[i/5].val() = porePressureY[i/5];
  //     fracturePressureY_AD[i/5].val() = fracturePressureY[i/5];
  //   }
  //
  //
  //   // Create vectors of empty AD types for the dependent variables
  //   vector<Sacado::Fad::DFad<double> > dilatation_AD(numEntries);
  //   vector<Sacado::Fad::DFad<double> > breaklessDilatation_AD(numEntries);
  //   vector<Sacado::Fad::DFad<double> > force_AD(3*numEntries);
  //   vector<Sacado::Fad::DFad<double> > phaseOnePoreFlow_AD(numEntries);
  //   vector<Sacado::Fad::DFad<double> > phaseOneFracFlow_AD(numEntries);
  //
  //   // Evaluate the constitutive model using the AD types
  //   MATERIAL_EVALUATION::computeBreaklessDilatation(x,&y_AD[0],weightedVolume,cellVolume,&breaklessDilatation_AD[0],&tempNeighborhoodList[0],tempNumOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
  //   MATERIAL_EVALUATION::computeDilatation(x,&y_AD[0],weightedVolume,cellVolume,bondDamage,&dilatation_AD[0],&tempNeighborhoodList[0],tempNumOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
  //   MATERIAL_EVALUATION::computeInternalForceLinearElasticCoupled(x,&y_AD[0],&porePressureY_AD[0],weightedVolume,cellVolume,&dilatation_AD[0],damage,bondDamage,scf,&force_AD[0],&tempNeighborhoodList[0],tempNumOwnedPoints,m_bulkModulus,m_shearModulus,m_horizon,m_alpha,deltaTemperature);
  //   MATERIAL_EVALUATION::computeInternalFlow(&y_AD[0],&porePressureY_AD[0],porePressureV,&fracturePressureY_AD[0],fracturePressureV,cellVolume,damage,criticalDilatation,&breaklessDilatation_AD[0],&phaseOnePoreFlow_AD[0],&phaseOneFracFlow_AD[0],&tempNeighborhoodList[0],tempNumOwnedPoints,m_permeabilityScalar,m_permeabilityCurveInflectionDamage,m_permeabilityAlpha,m_maxPermeability,m_phaseOneBasePerm,m_phaseOneDensity,m_phaseOneCompressibility,m_phaseOneViscosity,m_horizon,m_horizon_fracture,deltaTemperature);
  //
  //   // Load derivative values into scratch matrix
  //   // Multiply by volume along the way to convert force density to force
  //
  //   for(int row=0 ; row<numTotalNeighborhoodDof ; row+=5){
  //     for(int col=0 ; col<numTotalNeighborhoodDof ; col++){
  //       TEUCHOS_TEST_FOR_EXCEPT_MSG(!boost::math::isfinite(force_AD[row*3/5 + 0].dx(col)), "**** NaN detected in OnePhaseMultiphysicsElasticMaterial::computeAutomaticDifferentiationJacobian() ( fx ).\n");
  //       TEUCHOS_TEST_FOR_EXCEPT_MSG(!boost::math::isfinite(force_AD[row*3/5 + 1].dx(col)), "**** NaN detected in OnePhaseMultiphysicsElasticMaterial::computeAutomaticDifferentiationJacobian() ( fy ).\n");
  //       TEUCHOS_TEST_FOR_EXCEPT_MSG(!boost::math::isfinite(force_AD[row*3/5 + 2].dx(col)), "**** NaN detected in OnePhaseMultiphysicsElasticMaterial::computeAutomaticDifferentiationJacobian() ( fz ).\n");
  //       TEUCHOS_TEST_FOR_EXCEPT_MSG(!boost::math::isfinite(phaseOnePoreFlow_AD[row/5].dx(col)), "**** NaN detected in OnePhaseMultiphysicsElasticMaterial::computeAutomaticDifferentiationJacobian() (p1 pflw ).\n");
  //       TEUCHOS_TEST_FOR_EXCEPT_MSG(!boost::math::isfinite(phaseOneFracFlow_AD[row/5].dx(col)), "**** NaN detected in OnePhaseMultiphysicsElasticMaterial::computeAutomaticDifferentiationJacobian() (p1 fflw ).\n");
  //
  //       scratchMatrix(row + 0, col) = force_AD[row*3/5 + 0].dx(col) * cellVolume[row/5];
  //       scratchMatrix(row + 1, col) = force_AD[row*3/5 + 1].dx(col) * cellVolume[row/5];
  //       scratchMatrix(row + 2, col) = force_AD[row*3/5 + 2].dx(col) * cellVolume[row/5];
  //       scratchMatrix(row + 3, col) = phaseOnePoreFlow_AD[row/5].dx(col) * cellVolume[row/5];
  //       scratchMatrix(row + 4, col) = phaseOneFracFlow_AD[row/5].dx(col) * cellVolume[row/5];
  //     }
  //   }
  //
  //   // Sum the values into the global tangent matrix (this is expensive).
  //   if (jacobianType == PeridigmNS::Material::FULL_MATRIX)
  //     jacobian.addValues((int)globalIndices.size(), &globalIndices[0], scratchMatrix.Data());
  //   else if (jacobianType == PeridigmNS::Material::BLOCK_DIAGONAL) {
  //     jacobian.addBlockDiagonalValues((int)globalIndices.size(), &globalIndices[0], scratchMatrix.Data());
  //   }
  //   else // unknown jacobian type
  //     TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "**** Unknown Jacobian Type\n");
  // }

}

void PeridigmNS::OnePhaseMultiphysicsElasticMaterial::computeComplexStepFiniteDifferenceJacobian(const double dt,
                                                           const int numOwnedPoints,
                                                           const int* ownedIDs,
                                                           const int* neighborhoodList,
                                                           PeridigmNS::DataManager& dataManager,
                                                           PeridigmNS::SerialMatrix& jacobian,
                                                           PeridigmNS::Material::JacobianType jacobianType) const
{
  // The Jacobian is of the form:
  //
  // dF_0x/dx_0  dF_0x/dy_0  dF_0x/dz_0  dF_0x/dx_1  dF_0x/dy_1  dF_0x/dz_1  ...  dF_0x/dx_n  dF_0x/dy_n  dF_0x/dz_n
  // dF_0y/dx_0  dF_0y/dy_0  dF_0y/dz_0  dF_0y/dx_1  dF_0y/dy_1  dF_0y/dz_1  ...  dF_0y/dx_n  dF_0y/dy_n  dF_0y/dz_n
  // dF_0z/dx_0  dF_0z/dy_0  dF_0z/dz_0  dF_0z/dx_1  dF_0z/dy_1  dF_0z/dz_1  ...  dF_0z/dx_n  dF_0z/dy_n  dF_0z/dz_n
  // dF_1x/dx_0  dF_1x/dy_0  dF_1x/dz_0  dF_1x/dx_1  dF_1x/dy_1  dF_1x/dz_1  ...  dF_1x/dx_n  dF_1x/dy_n  dF_1x/dz_n
  // dF_1y/dx_0  dF_1y/dy_0  dF_1y/dz_0  dF_1y/dx_1  dF_1y/dy_1  dF_1y/dz_1  ...  dF_1y/dx_n  dF_1y/dy_n  dF_1y/dz_n
  // dF_1z/dx_0  dF_1z/dy_0  dF_1z/dz_0  dF_1z/dx_1  dF_1z/dy_1  dF_1z/dz_1  ...  dF_1z/dx_n  dF_1z/dy_n  dF_1z/dz_n
  //     .           .           .           .           .           .                .           .           .
  //     .           .           .           .           .           .                .           .           .
  //     .           .           .           .           .           .                .           .           .
  // dF_nx/dx_0  dF_nx/dy_0  dF_nx/dz_0  dF_nx/dx_1  dF_nx/dy_1  dF_nx/dz_1  ...  dF_nx/dx_n  dF_nx/dy_n  dF_nx/dz_n
  // dF_ny/dx_0  dF_ny/dy_0  dF_ny/dz_0  dF_ny/dx_1  dF_ny/dy_1  dF_ny/dz_1  ...  dF_ny/dx_n  dF_ny/dy_n  dF_ny/dz_n
  // dF_nz/dx_0  dF_nz/dy_0  dF_nz/dz_0  dF_nz/dx_1  dF_nz/dy_1  dF_nz/dz_1  ...  dF_nz/dx_n  dF_nz/dy_n  dF_nz/dz_n

  // Each entry is computed by finite difference:
  //
  // Forward difference:
  // dF_0x/dx_0 = ( F_0x(perturbed x_0) - F_0x(unperturbed) ) / epsilon
  //
  // Central difference:
  // dF_0x/dx_0 = ( F_0x(positive perturbed x_0) - F_0x(negative perturbed x_0) ) / ( 2.0*epsilon )
  //
  // Complex Step:
  // dF_0x/dx_0 = imag[ ( F_0x(x_0 + i*epsilon) ) / epsilon ]

  TEUCHOS_TEST_FOR_EXCEPT_MSG(m_finiteDifferenceProbeLength == DBL_MAX, "**** Finite-difference Jacobian requires that the \"Finite Difference Probe Length\" parameter be set.\n");
  double epsilon = m_finiteDifferenceProbeLength;

  //These are declared static in order to prevent the slowdown from re-allocation
  //Because they are visited for every perturbation, there is a chance the cache update policy
  //will cause these values to remain in cache while the Jacbobian is being computed.
  static vector<std::complex<double> > forceComplex;
  static vector<std::complex<double> > yComplex;
  static vector<std::complex<double> > porePressureYComplex;
  static vector<std::complex<double> > porePressureVComplex;
  static vector<std::complex<double> > fracturePressureYComplex;
  static vector<std::complex<double> > fracturePressureVComplex;
  static vector<std::complex<double> > dilatationComplex;
  static vector<std::complex<double> > breaklessDilatationComplex;
  static vector<std::complex<double> > phaseOnePoreFlowComplex;
  static vector<std::complex<double> > phaseOneFracFlowComplex;
  static vector<std::complex<double> > matrixPorosityComplex;
  static vector<std::complex<double> > fracturePorosityComplex;
  static vector<std::complex<double> > phaseOneDensityInPoresComplex;
  static vector<std::complex<double> > phaseOneDensityInFractureComplex;

  // Loop over all points.
  int neighborhoodListIndex = 0;
  for(int iID=0 ; iID<numOwnedPoints ; ++iID){

    // Create a temporary neighborhood consisting of a single point and its neighbors.
    int numNeighbors = neighborhoodList[neighborhoodListIndex++];
    int numEntries = numNeighbors+1;

    vector<int> tempMyGlobalIDs(numNeighbors+1);
    // Put the node at the center of the neighborhood at the beginning of the list.
    tempMyGlobalIDs[0] = dataManager.getOwnedScalarPointMap()->GID(iID);
    vector<int> tempNeighborhoodList(numNeighbors+1);
    tempNeighborhoodList[0] = numNeighbors;
    for(int iNID=0 ; iNID<numNeighbors ; ++iNID){
      int neighborID = neighborhoodList[neighborhoodListIndex++];
      tempMyGlobalIDs[iNID+1] = dataManager.getOverlapScalarPointMap()->GID(neighborID);
      tempNeighborhoodList[iNID+1] = iNID+1;
    }

    Epetra_SerialComm serialComm;
    Teuchos::RCP<Epetra_BlockMap> tempOneDimensionalMap = Teuchos::rcp(new Epetra_BlockMap(numNeighbors+1, numNeighbors+1, &tempMyGlobalIDs[0], 1, 0, serialComm));
    Teuchos::RCP<Epetra_BlockMap> tempThreeDimensionalMap = Teuchos::rcp(new Epetra_BlockMap(numNeighbors+1, numNeighbors+1, &tempMyGlobalIDs[0], 3, 0, serialComm));
    Teuchos::RCP<Epetra_BlockMap> tempBondMap = Teuchos::rcp(new Epetra_BlockMap(1, 1, &tempMyGlobalIDs[0], numNeighbors, 0, serialComm));

    // Create a temporary DataManager containing data for this point and its neighborhood.
    PeridigmNS::DataManager tempDataManager;
    tempDataManager.setMaps(Teuchos::RCP<const Epetra_BlockMap>(),
                            tempOneDimensionalMap,
                            Teuchos::RCP<const Epetra_BlockMap>(),
                            tempThreeDimensionalMap,
                            tempBondMap);

    // The temporary data manager will have the same fields and data as the real data manager.
    vector<int> fieldIds = dataManager.getFieldIds();
    tempDataManager.allocateData(fieldIds);
    tempDataManager.copyLocallyOwnedDataFromDataManager(dataManager);

    // Set up numOwnedPoints and ownedIDs.
    // There is only one owned ID, and it has local ID zero in the tempDataManager.
    int tempNumOwnedPoints = 1;
    vector<int> tempOwnedIDs(1);
    tempOwnedIDs[0] = 0;

    // Extract pointers to the underlying data
    double *x, *y, *cellVolume, *weightedVolume;
    double *dilatationN, *criticalDilatation;
    double *bondDamage, *damage, *force, *scf, *deltaTemperature;
    double *phaseOneDensityInPores, *phaseOneDensityInFracture, *phaseOneDensityInPoresN, *phaseOneDensityInFractureN;
    double *matrixPorosityN, *fracturePorosityN;
    double *porePressureYN, *fracturePressureYN, *porePressureYNP1, *fracturePressureYNP1, *porePressureVNP1, *fracturePressureVNP1;

    tempDataManager.getData(m_modelCoordinatesFieldId, PeridigmField::STEP_NONE)->ExtractView(&x);
    tempDataManager.getData(m_coordinatesFieldId, PeridigmField::STEP_NP1)->ExtractView(&y);
    tempDataManager.getData(m_volumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&cellVolume);
    tempDataManager.getData(m_weightedVolumeFieldId, PeridigmField::STEP_NONE)->ExtractView(&weightedVolume);
    tempDataManager.getData(m_dilatationFieldId, PeridigmField::STEP_N)->ExtractView(&dilatationN);
    tempDataManager.getData(m_criticalDilatationFieldId, PeridigmField::STEP_NONE)->ExtractView(&criticalDilatation);
    tempDataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_NP1)->ExtractView(&bondDamage);
    tempDataManager.getData(m_damageFieldId, PeridigmField::STEP_NP1)->ExtractView(&damage);
    tempDataManager.getData(m_surfaceCorrectionFactorFieldId, PeridigmField::STEP_NONE)->ExtractView(&scf);
    tempDataManager.getData(m_porePressureYFieldId, PeridigmField::STEP_N)->ExtractView(&porePressureYN);
    tempDataManager.getData(m_fracturePressureYFieldId, PeridigmField::STEP_N)->ExtractView(&fracturePressureYN);
    tempDataManager.getData(m_porePressureYFieldId, PeridigmField::STEP_NP1)->ExtractView(&porePressureYNP1);
    tempDataManager.getData(m_fracturePressureYFieldId, PeridigmField::STEP_NP1)->ExtractView(&fracturePressureYNP1);
    tempDataManager.getData(m_porePressureVFieldId, PeridigmField::STEP_NP1)->ExtractView(&porePressureVNP1);
    tempDataManager.getData(m_fracturePressureVFieldId, PeridigmField::STEP_NP1)->ExtractView(&fracturePressureVNP1);
    tempDataManager.getData(m_phaseOneDensityInPoresFieldId, PeridigmField::STEP_N)->ExtractView(&phaseOneDensityInPoresN);
    tempDataManager.getData(m_phaseOneDensityInFractureFieldId, PeridigmField::STEP_N)->ExtractView(&phaseOneDensityInFractureN);
    tempDataManager.getData(m_matrixPorosityFieldId, PeridigmField::STEP_N)->ExtractView(&matrixPorosityN);
    tempDataManager.getData(m_fracturePorosityFieldId, PeridigmField::STEP_N)->ExtractView(&fracturePorosityN);

    deltaTemperature = NULL;
    if(m_applyThermalStrains) tempDataManager.getData(m_deltaTemperatureFieldId, PeridigmField::STEP_NP1)->ExtractView(&deltaTemperature);


    // Resize the temporary vectors
    if(yComplex.size() < (3*numEntries)){
      forceComplex.resize(3*numEntries);
      yComplex.resize(3*numEntries);
      porePressureYComplex.resize(numEntries);
      porePressureVComplex.resize(numEntries);
      fracturePressureYComplex.resize(numEntries);
      fracturePressureVComplex.resize(numEntries);
      dilatationComplex.resize(numEntries);
      breaklessDilatationComplex.resize(numEntries);
      phaseOnePoreFlowComplex.resize(numEntries);
      phaseOneFracFlowComplex.resize(numEntries);
      matrixPorosityComplex.resize(numEntries);
      fracturePorosityComplex.resize(numEntries);
      phaseOneDensityInPoresComplex.resize(numEntries);
      phaseOneDensityInFractureComplex.resize(numEntries);
    }

    // Reset dependent temporary variables to zero as if computeForce were called
    for(int i=0; i<numEntries; ++i){
      forceComplex[3*i + 0] = 0.0;
      forceComplex[3*i + 1] = 0.0;
      forceComplex[3*i + 2] = 0.0;
      phaseOnePoreFlowComplex[i] = 0.0;
      phaseOneFracFlowComplex[i] = 0.0;
    }

    // Set the values for the independent temporary variables
    for(int i=0 ; i< 5*numEntries ; i+=5){
      for(int j=0 ; j<3 ; ++j){
        yComplex[i*3/5+j] =  std::complex<double>(y[i*3/5+j], 0.0);
      }
      porePressureYComplex[i/5] = std::complex<double>(porePressureYNP1[i/5], 0.0);
      porePressureVComplex[i/5] = std::complex<double>(porePressureVNP1[i/5], 0.0);
      fracturePressureYComplex[i/5] = std::complex<double>(fracturePressureYNP1[i/5], 0.0);
      fracturePressureVComplex[i/5] = std::complex<double>(fracturePressureVNP1[i/5], 0.0);
    }

    // Use the scratchMatrix as sub-matrix for storing tangent values prior to loading them into the global tangent matrix.
    // Resize scratchMatrix if necessary
    if(scratchMatrix.Dimension() < 5*(numNeighbors+1))
      scratchMatrix.Resize(5*(numNeighbors+1));

    // Create a list of global indices for the rows/columns in the scratch matrix.
    vector<int> globalIndices(5*(numNeighbors+1));
    for(int i=0 ; i<numNeighbors+1 ; ++i){
      int globalID = tempOneDimensionalMap->GID(i);
      for(int j=0 ; j<5 ; ++j)
        globalIndices[5*i+j] = 5*globalID+j;
    }

    // Perturb one dof in the neighborhood at a time and compute the constitutive model.
    // The point itself plus each of its neighbors must be perturbed.
    for(int iNID=0 ; iNID<numNeighbors+1 ; ++iNID){

      int perturbID; //perturbID is the node we are perturbing
      if(iNID < numNeighbors)
        perturbID = tempNeighborhoodList[iNID+1];
      else
        perturbID = 0;

      for(int dof=0 ; dof<3 ; ++dof){
        // Perturb a dof and evaluate the model.
        yComplex[3*perturbID+dof] += std::complex<double>(0.0, epsilon);
        //TODO if there is a viscous effect term, we need to perturb velocity alongside current configuration
        // Evaluate the model
        MATERIAL_EVALUATION::computeBreaklessDilatation(x,&yComplex[0],weightedVolume,cellVolume,&breaklessDilatationComplex[0],neighborhoodList,numOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
        MATERIAL_EVALUATION::computeDilatation(x,&yComplex[0],weightedVolume,cellVolume,bondDamage,&dilatationComplex[0],neighborhoodList,numOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
        MATERIAL_EVALUATION::computeMatrixPorosity(&matrixPorosityComplex[0],matrixPorosityN,&porePressureYComplex[0],porePressureYN,&dilatationComplex[0],dilatationN,m_compressibilityRock,m_alphaBiot,numOwnedPoints);
        MATERIAL_EVALUATION::computeFracturePorosityComplex(&fracturePorosityComplex[0],&breaklessDilatationComplex[0],criticalDilatation,numOwnedPoints);
        MATERIAL_EVALUATION::computePhaseOneDensityInPores(&phaseOneDensityInPoresComplex[0],&porePressureYComplex[0],deltaTemperature,numOwnedPoints);
        MATERIAL_EVALUATION::computePhaseOneDensityInFracture(&phaseOneDensityInFractureComplex[0],&fracturePressureYComplex[0],deltaTemperature,numOwnedPoints);
        MATERIAL_EVALUATION::computeInternalForceLinearElasticCoupled(x,&yComplex[0],&porePressureYComplex[0],&fracturePressureYComplex[0],weightedVolume,cellVolume,&dilatationComplex[0],damage,bondDamage,scf,&forceComplex[0],neighborhoodList,numOwnedPoints,m_bulkModulus,m_shearModulus,m_horizon,m_alphaBiot,m_alpha,deltaTemperature);
        MATERIAL_EVALUATION::ONE_PHASE::computeInternalFlowComplex(&yComplex[0],&porePressureYComplex[0],&porePressureVComplex[0],&fracturePressureYComplex[0],&fracturePressureVComplex[0],cellVolume,damage,&matrixPorosityComplex[0],matrixPorosityN,&fracturePorosityComplex[0],fracturePorosityN,&phaseOneDensityInPoresComplex[0],phaseOneDensityInPoresN,&phaseOneDensityInFractureComplex[0],phaseOneDensityInFractureN,&breaklessDilatationComplex[0],&phaseOnePoreFlowComplex[0],&phaseOneFracFlowComplex[0],neighborhoodList,numOwnedPoints,m_matrixPermeabilityXX,m_matrixPermeabilityYY,m_matrixPermeabilityZZ,m_phaseOneViscosity,m_horizon,m_horizon_fracture,dt,deltaTemperature);
        // Restore unperturbed value
        yComplex[3*perturbID+dof] = std::real(yComplex[3*perturbID+dof]);
        // Enter derivatives into a buffer before transferring them to the Jacobian
        for(int i=0 ; i<numNeighbors+1 ; ++i){
          int dependentID;
          if(i < numNeighbors)
            dependentID = tempNeighborhoodList[i+1];
          else
            dependentID = 0;
          scratchMatrix(5*dependentID+0, 5*perturbID+dof) = std::imag( forceComplex[3*dependentID + 0]/epsilon );
          scratchMatrix(5*dependentID+1, 5*perturbID+dof) = std::imag( forceComplex[3*dependentID + 1]/epsilon );
          scratchMatrix(5*dependentID+2, 5*perturbID+dof) = std::imag( forceComplex[3*dependentID + 2]/epsilon );
          scratchMatrix(5*dependentID+3, 5*perturbID+dof) = std::imag( phaseOnePoreFlowComplex[dependentID]/epsilon );
          scratchMatrix(5*dependentID+4, 5*perturbID+dof) = std::imag( phaseOneFracFlowComplex[dependentID]/epsilon );
          // Reset dependent variables to zero as if computeForce were called
          forceComplex[3*dependentID + 0] = 0.0;
          forceComplex[3*dependentID + 1] = 0.0;
          forceComplex[3*dependentID + 2] = 0.0;
          phaseOnePoreFlowComplex[dependentID] = 0.0;
          phaseOneFracFlowComplex[dependentID] = 0.0;
        }
      }

      //dof == 3
      // Perturb a dof and evaluate the model.
      porePressureYComplex[perturbID] += std::complex<double>(0.0, epsilon);
      porePressureVComplex[perturbID] += std::complex<double>(0.0, epsilon/dt);
      // Evaluate the constitutive model
      MATERIAL_EVALUATION::computeBreaklessDilatation(x,&yComplex[0],weightedVolume,cellVolume,&breaklessDilatationComplex[0],neighborhoodList,numOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
      MATERIAL_EVALUATION::computeDilatation(x,&yComplex[0],weightedVolume,cellVolume,bondDamage,&dilatationComplex[0],neighborhoodList,numOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
      MATERIAL_EVALUATION::computeMatrixPorosity(&matrixPorosityComplex[0],matrixPorosityN,&porePressureYComplex[0],porePressureYN,&dilatationComplex[0],dilatationN,m_compressibilityRock,m_alphaBiot,numOwnedPoints);
      MATERIAL_EVALUATION::computeFracturePorosityComplex(&fracturePorosityComplex[0],&breaklessDilatationComplex[0],criticalDilatation,numOwnedPoints);
      MATERIAL_EVALUATION::computePhaseOneDensityInPores(&phaseOneDensityInPoresComplex[0],&porePressureYComplex[0],deltaTemperature,numOwnedPoints);
      MATERIAL_EVALUATION::computePhaseOneDensityInFracture(&phaseOneDensityInFractureComplex[0],&fracturePressureYComplex[0],deltaTemperature,numOwnedPoints);
      MATERIAL_EVALUATION::computeInternalForceLinearElasticCoupled(x,&yComplex[0],&porePressureYComplex[0],&fracturePressureYComplex[0],weightedVolume,cellVolume,&dilatationComplex[0],damage,bondDamage,scf,&forceComplex[0],neighborhoodList,numOwnedPoints,m_bulkModulus,m_shearModulus,m_horizon,m_alphaBiot,m_alpha,deltaTemperature);
      MATERIAL_EVALUATION::ONE_PHASE::computeInternalFlowComplex(&yComplex[0],&porePressureYComplex[0],&porePressureVComplex[0],&fracturePressureYComplex[0],&fracturePressureVComplex[0],cellVolume,damage,&matrixPorosityComplex[0],matrixPorosityN,&fracturePorosityComplex[0],fracturePorosityN,&phaseOneDensityInPoresComplex[0],phaseOneDensityInPoresN,&phaseOneDensityInFractureComplex[0],phaseOneDensityInFractureN,&breaklessDilatationComplex[0],&phaseOnePoreFlowComplex[0],&phaseOneFracFlowComplex[0],neighborhoodList,numOwnedPoints,m_matrixPermeabilityXX,m_matrixPermeabilityYY,m_matrixPermeabilityZZ,m_phaseOneViscosity,m_horizon,m_horizon_fracture,dt,deltaTemperature);
      // Restore unperturbed value
      porePressureYComplex[perturbID] = std::real(porePressureYComplex[perturbID]);
      porePressureVComplex[perturbID] = std::real(porePressureVComplex[perturbID]);
      // Enter derivatives into a buffer before transferring them to the Jacobian
      for(int i=0 ; i<numNeighbors+1 ; ++i){
        int dependentID;
        if(i < numNeighbors)
          dependentID = tempNeighborhoodList[i+1];
        else
          dependentID = 0;
        scratchMatrix(5*dependentID+0, 5*perturbID+3) = std::imag( forceComplex[3*dependentID + 0]/epsilon );
        scratchMatrix(5*dependentID+1, 5*perturbID+3) = std::imag( forceComplex[3*dependentID + 1]/epsilon );
        scratchMatrix(5*dependentID+2, 5*perturbID+3) = std::imag( forceComplex[3*dependentID + 2]/epsilon );
        scratchMatrix(5*dependentID+3, 5*perturbID+3) = std::imag( phaseOnePoreFlowComplex[dependentID]/epsilon );
        scratchMatrix(5*dependentID+4, 5*perturbID+3) = std::imag( phaseOneFracFlowComplex[dependentID]/epsilon );
        // Reset dependent variables to zero as if computeForce were called
        forceComplex[3*dependentID + 0] = 0.0;
        forceComplex[3*dependentID + 1] = 0.0;
        forceComplex[3*dependentID + 2] = 0.0;
        phaseOnePoreFlowComplex[dependentID] = 0.0;
        phaseOneFracFlowComplex[dependentID] = 0.0;
      }

      //dof == 4
      // Perturb a dof and evaluate the model.
      fracturePressureYComplex[perturbID] += std::complex<double>(0.0, epsilon);
      fracturePressureVComplex[perturbID] += std::complex<double>(0.0, epsilon/dt);
      // Evaluate the constitutive model
      MATERIAL_EVALUATION::computeBreaklessDilatation(x,&yComplex[0],weightedVolume,cellVolume,&breaklessDilatationComplex[0],neighborhoodList,numOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
      MATERIAL_EVALUATION::computeDilatation(x,&yComplex[0],weightedVolume,cellVolume,bondDamage,&dilatationComplex[0],neighborhoodList,numOwnedPoints,m_horizon,m_OMEGA,m_alpha,deltaTemperature);
      MATERIAL_EVALUATION::computeMatrixPorosity(&matrixPorosityComplex[0],matrixPorosityN,&porePressureYComplex[0],porePressureYN,&dilatationComplex[0],dilatationN,m_compressibilityRock,m_alphaBiot,numOwnedPoints);
      MATERIAL_EVALUATION::computeFracturePorosityComplex(&fracturePorosityComplex[0],&breaklessDilatationComplex[0],criticalDilatation,numOwnedPoints);
      MATERIAL_EVALUATION::computePhaseOneDensityInPores(&phaseOneDensityInPoresComplex[0],&porePressureYComplex[0],deltaTemperature,numOwnedPoints);
      MATERIAL_EVALUATION::computePhaseOneDensityInFracture(&phaseOneDensityInFractureComplex[0],&fracturePressureYComplex[0],deltaTemperature,numOwnedPoints);
      MATERIAL_EVALUATION::computeInternalForceLinearElasticCoupled(x,&yComplex[0],&porePressureYComplex[0],&fracturePressureYComplex[0],weightedVolume,cellVolume,&dilatationComplex[0],damage,bondDamage,scf,&forceComplex[0],neighborhoodList,numOwnedPoints,m_bulkModulus,m_shearModulus,m_horizon,m_alphaBiot,m_alpha,deltaTemperature);
      MATERIAL_EVALUATION::ONE_PHASE::computeInternalFlowComplex(&yComplex[0],&porePressureYComplex[0],&porePressureVComplex[0],&fracturePressureYComplex[0],&fracturePressureVComplex[0],cellVolume,damage,&matrixPorosityComplex[0],matrixPorosityN,&fracturePorosityComplex[0],fracturePorosityN,&phaseOneDensityInPoresComplex[0],phaseOneDensityInPoresN,&phaseOneDensityInFractureComplex[0],phaseOneDensityInFractureN,&breaklessDilatationComplex[0],&phaseOnePoreFlowComplex[0],&phaseOneFracFlowComplex[0],neighborhoodList,numOwnedPoints,m_matrixPermeabilityXX,m_matrixPermeabilityYY,m_matrixPermeabilityZZ,m_phaseOneViscosity,m_horizon,m_horizon_fracture,dt,deltaTemperature);
      // Restore unperturbed value
      fracturePressureYComplex[perturbID] = std::real(fracturePressureYComplex[perturbID]);
      fracturePressureVComplex[perturbID] = std::real(fracturePressureVComplex[perturbID]);
      // Enter derivatives into a buffer before transferring them to the Jacobian
      for(int i=0 ; i<numNeighbors+1 ; ++i){
        int dependentID;
        if(i < numNeighbors)
          dependentID = tempNeighborhoodList[i+1];
        else
          dependentID = 0;
        scratchMatrix(5*dependentID+0, 5*perturbID+4) = std::imag( forceComplex[3*dependentID + 0]/epsilon );
        scratchMatrix(5*dependentID+1, 5*perturbID+4) = std::imag( forceComplex[3*dependentID + 1]/epsilon );
        scratchMatrix(5*dependentID+2, 5*perturbID+4) = std::imag( forceComplex[3*dependentID + 2]/epsilon );
        scratchMatrix(5*dependentID+3, 5*perturbID+4) = std::imag( phaseOnePoreFlowComplex[dependentID]/epsilon );
        scratchMatrix(5*dependentID+4, 5*perturbID+4) = std::imag( phaseOneFracFlowComplex[dependentID]/epsilon );
        // Reset dependent variables to zero as if computeForce were called
        forceComplex[3*dependentID + 0] = 0.0;
        forceComplex[3*dependentID + 1] = 0.0;
        forceComplex[3*dependentID + 2] = 0.0;
        phaseOnePoreFlowComplex[dependentID] = 0.0;
        phaseOneFracFlowComplex[dependentID] = 0.0;
      }
    }

    // Convert force density to force
    // \todo Create utility function for this in ScratchMatrix
    for(unsigned int row=0 ; row<globalIndices.size() ; ++row){
      for(unsigned int col=0 ; col<globalIndices.size() ; col+=5){
        scratchMatrix(row, col+0) *= cellVolume[row/5];
        scratchMatrix(row, col+1) *= cellVolume[row/5];
        scratchMatrix(row, col+2) *= cellVolume[row/5];
        scratchMatrix(row, col+3) *= cellVolume[row/5];
        scratchMatrix(row, col+4) *= cellVolume[row/5];
      }
    }

    // Check for NaNs
    for(unsigned int row=0 ; row<globalIndices.size() ; ++row){
      for(unsigned int col=0 ; col<globalIndices.size() ; ++col){
        TEUCHOS_TEST_FOR_EXCEPT_MSG(!boost::math::isfinite(scratchMatrix(row, col)), "**** NaN detected in finite-difference Jacobian.\n");
      }
    }

    // Sum the values into the global tangent matrix (this is expensive).
    if (jacobianType == PeridigmNS::Material::FULL_MATRIX)
      jacobian.addValues((int)globalIndices.size(), &globalIndices[0], scratchMatrix.Data());
    else if (jacobianType == PeridigmNS::Material::BLOCK_DIAGONAL) {
      jacobian.addBlockDiagonalValues((int)globalIndices.size(), &globalIndices[0], scratchMatrix.Data());
    }
    else // unknown jacobian type
      TEUCHOS_TEST_FOR_EXCEPT_MSG(true, "**** Unknown Jacobian Type\n");
  }
}
