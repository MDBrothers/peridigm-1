/*! \file Peridigm_FractureSpaceCriticalStretchDamageModel.cpp */

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

#include "Peridigm_FractureSpaceCriticalStretchDamageModel.hpp"
#include "Peridigm_Field.hpp"

using namespace std;

PeridigmNS::FractureSpaceCriticalStretchDamageModel::FractureSpaceCriticalStretchDamageModel(const Teuchos::ParameterList& params)
  : DamageModel(params), m_applyThermalStrains(false), m_modelCoordinatesFieldId(-1), m_coordinatesFieldId(-1), m_damageFieldId(-1), m_bondDamageFieldId(-1), m_deltaTemperatureFieldId(-1)
{
  m_criticalStretch = params.get<double>("Critical Stretch");

  if(params.isParameter("Thermal Expansion Coefficient")){
    m_alpha = params.get<double>("Thermal Expansion Coefficient");
    m_applyThermalStrains = true;
  }

  if(params.isSublist("Multiphysics Bond Filters")){
    myParams = Teuchos::RCP<Teuchos::ParameterList>( new Teuchos::ParameterList(params.sublist("Multiphysics Bond Filters")) );
    m_initialBondFiltersPresent = true;
  }
  else
    m_initialBondFiltersPresent = false;

  PeridigmNS::FieldManager& fieldManager = PeridigmNS::FieldManager::self();
  m_modelCoordinatesFieldId = fieldManager.getFieldId("Model_Coordinates");
  m_coordinatesFieldId = fieldManager.getFieldId("Coordinates");
  m_damageFieldId = fieldManager.getFieldId(PeridigmNS::PeridigmField::ELEMENT, PeridigmNS::PeridigmField::SCALAR, PeridigmNS::PeridigmField::TWO_STEP, "Damage");
  m_bondDamageFieldId = fieldManager.getFieldId(PeridigmNS::PeridigmField::BOND, PeridigmNS::PeridigmField::SCALAR, PeridigmNS::PeridigmField::TWO_STEP, "Bond_Damage");
  //m_fractureConnectedFieldId = fieldManager.getFieldId(PeridigmField::ELEMENT, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Fracture_Connected");

  if(m_applyThermalStrains)
    m_deltaTemperatureFieldId = fieldManager.getFieldId(PeridigmField::NODE, PeridigmField::SCALAR, PeridigmField::TWO_STEP, "Temperature_Change");

  m_fieldIds.push_back(m_modelCoordinatesFieldId);
  m_fieldIds.push_back(m_coordinatesFieldId);
  m_fieldIds.push_back(m_damageFieldId);
  m_fieldIds.push_back(m_bondDamageFieldId);
  if(m_applyThermalStrains)
    m_fieldIds.push_back(m_deltaTemperatureFieldId);


}

PeridigmNS::FractureSpaceCriticalStretchDamageModel::~FractureSpaceCriticalStretchDamageModel()
{
}

void
PeridigmNS::FractureSpaceCriticalStretchDamageModel::initialize(const double dt,
                                                   const int numOwnedPoints,
                                                   const int* ownedIDs,
                                                   const int* neighborhoodList,
                                                   PeridigmNS::DataManager& dataManager) const
{
  double *previousDamage, *damage, *bondDamage, *previousBondDamage, *x;
  dataManager.getData(m_damageFieldId, PeridigmField::STEP_N)->ExtractView(&previousDamage);
  dataManager.getData(m_damageFieldId, PeridigmField::STEP_NP1)->ExtractView(&damage);
  dataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_N)->ExtractView(&previousBondDamage);
  dataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_NP1)->ExtractView(&bondDamage);

  // Get positional data for bond filtering too
  dataManager.getData(m_modelCoordinatesFieldId, PeridigmField::STEP_NONE)->ExtractView(&x);

  PdBondFilter::FinitePlane* filterPlanes;

  if(m_initialBondFiltersPresent){
    for (Teuchos::ParameterList::ConstIterator it = myParams->begin(); it != myParams->end(); ++it) {
      string parameterListName = it->first;
      Teuchos::ParameterList params = myParams->sublist(parameterListName);
      string type = params.get<string>("Type");
      if(type == "Rectangular_Plane"){
        double normal[3], lowerLeftCorner[3], bottomUnitVector[3], bottomLength, sideLength;
        normal[0] = params.get<double>("Normal_X");
        normal[1] = params.get<double>("Normal_Y");
        normal[2] = params.get<double>("Normal_Z");
        lowerLeftCorner[0] = params.get<double>("Lower_Left_Corner_X");
        lowerLeftCorner[1] = params.get<double>("Lower_Left_Corner_Y");
        lowerLeftCorner[2] = params.get<double>("Lower_Left_Corner_Z");
        bottomUnitVector[0] = params.get<double>("Bottom_Unit_Vector_X");
        bottomUnitVector[1] = params.get<double>("Bottom_Unit_Vector_Y");
        bottomUnitVector[2] = params.get<double>("Bottom_Unit_Vector_Z");
        bottomLength = params.get<double>("Bottom_Length");
        sideLength = params.get<double>("Side_Length");
        PdBondFilter::FinitePlane finitePlane(normal, lowerLeftCorner, bottomUnitVector, bottomLength, sideLength);
        filterPlanes = new PdBondFilter::FinitePlane(finitePlane);
      }
      else{
        string msg = "\n**** Error, invalid bond filter type:  " + type;
        msg += "\n**** Allowable types are:  Rectangular_Plane\n";
        TEUCHOS_TEST_FOR_EXCEPT_MSG(true, msg);
      }
    }
  }
  // Initialize bond damage to zero if bond does not cross a filter plane
  // Specify the head of the bond as neighborX and the tail as ownedX
  double ownedX[3], neighborX[3], intersectionLocation[3];
  double t;
  const int * neighPtr = neighborhoodList;

  int neighborID(0), bondIndex(0), iID(0), iNID(0);
  for(int iID=0 ; iID<numOwnedPoints ; ++iID){
    int nodeID = ownedIDs[iID];
    int numNeighbors = *neighPtr; neighPtr++;

    ownedX[0] = x[nodeID*3];
    ownedX[1] = x[nodeID*3+1];
    ownedX[2] = x[nodeID*3+2];

    for(int iNID=0 ; iNID<numNeighbors ; ++iNID, neighPtr++, bondIndex++){
      neighborID = *neighPtr;
      neighborX[0] = x[3*neighborID];
      neighborX[1] = x[3*neighborID + 1];
      neighborX[2] = x[3*neighborID + 2];

      if(m_initialBondFiltersPresent){
        if(0 != filterPlanes->bondIntersectInfinitePlane(&ownedX[0], &neighborX[0], t, &intersectionLocation[0]) && filterPlanes->bondIntersect(&intersectionLocation[0]) ){
          bondDamage[bondIndex] = 1.00;
          previousBondDamage[bondIndex] = 1.00;
        }
        else{
          bondDamage[bondIndex] = 0.0;
          previousBondDamage[bondIndex] = 0.0;
        }
      }
      else{
        bondDamage[bondIndex] = 0.0;
        previousBondDamage[bondIndex] = 0.0;
      }
    }
  }

  //  Update the element damage (percent of bonds broken)

  int neighborhoodListIndex = 0;
  bondIndex = 0;
  for(iID=0 ; iID<numOwnedPoints ; ++iID){
    int nodeId = ownedIDs[iID];
    int numNeighbors = neighborhoodList[neighborhoodListIndex++];
    neighborhoodListIndex += numNeighbors;
    double totalDamage = 0.0;
    for(iNID=0 ; iNID<numNeighbors ; ++iNID){
      totalDamage += bondDamage[bondIndex++];
    }
    if(numNeighbors > 0)
      totalDamage /= numNeighbors;
    else
      totalDamage = 0.0;
    damage[nodeId] = totalDamage;
    previousDamage[nodeId] = totalDamage;
  }

}

void
PeridigmNS::FractureSpaceCriticalStretchDamageModel::computeDamage(const double dt,
                                                      const int numOwnedPoints,
                                                      const int* ownedIDs,
                                                      const int* neighborhoodList,
                                                      PeridigmNS::DataManager& dataManager) const
{
  double *x, *y, *damage, *bondDamageNP1, *deltaTemperature;
  dataManager.getData(m_modelCoordinatesFieldId, PeridigmField::STEP_NONE)->ExtractView(&x);
  dataManager.getData(m_coordinatesFieldId, PeridigmField::STEP_NP1)->ExtractView(&y);
  dataManager.getData(m_damageFieldId, PeridigmField::STEP_NP1)->ExtractView(&damage);
  dataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_NP1)->ExtractView(&bondDamageNP1);
  deltaTemperature = NULL;
  if(m_applyThermalStrains)
    dataManager.getData(m_deltaTemperatureFieldId, PeridigmField::STEP_NP1)->ExtractView(&deltaTemperature);

  double trialDamage(0.0);
  int neighborhoodListIndex(0), bondIndex(0);
  int nodeId, numNeighbors, neighborID, iID, iNID;
  double nodeInitialX[3], nodeCurrentX[3], initialDistance, currentDistance, relativeExtension, totalDamage;

  // Set the bond damage to the previous value
  *(dataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_NP1)) = *(dataManager.getData(m_bondDamageFieldId, PeridigmField::STEP_N));

  // Update the bond damage
  // Break bonds if the extension is greater than the critical extension
  unsigned new_bonds_broken = 0;
  for(iID=0 ; iID<numOwnedPoints ; ++iID){
  nodeId = ownedIDs[iID];
  nodeInitialX[0] = x[nodeId*3];
  nodeInitialX[1] = x[nodeId*3+1];
  nodeInitialX[2] = x[nodeId*3+2];
  nodeCurrentX[0] = y[nodeId*3];
  nodeCurrentX[1] = y[nodeId*3+1];
  nodeCurrentX[2] = y[nodeId*3+2];
  numNeighbors = neighborhoodList[neighborhoodListIndex++];
  for(iNID=0 ; iNID<numNeighbors ; ++iNID){
    neighborID = neighborhoodList[neighborhoodListIndex++];
      initialDistance =
        distance(nodeInitialX[0], nodeInitialX[1], nodeInitialX[2],
                 x[neighborID*3], x[neighborID*3+1], x[neighborID*3+2]);
      currentDistance =
        distance(nodeCurrentX[0], nodeCurrentX[1], nodeCurrentX[2],
                 y[neighborID*3], y[neighborID*3+1], y[neighborID*3+2]);
      if(m_applyThermalStrains)
        currentDistance -= m_alpha*deltaTemperature[nodeId]*initialDistance;
      relativeExtension = (currentDistance - initialDistance)/initialDistance;
      trialDamage = 0.0;
      if(relativeExtension > m_criticalStretch)
        trialDamage = 1.0;
      if(trialDamage > bondDamageNP1[bondIndex]){
        bondDamageNP1[bondIndex] = trialDamage;
        new_bonds_broken++;
      }
      bondIndex += 1;
    }
  }

  //  Update the element damage (percent of bonds broken)
  neighborhoodListIndex = 0;
  bondIndex = 0;
  for(iID=0 ; iID<numOwnedPoints ; ++iID){
    nodeId = ownedIDs[iID];
    numNeighbors = neighborhoodList[neighborhoodListIndex++];
    neighborhoodListIndex += numNeighbors;
    totalDamage = 0.0;
    for(iNID=0 ; iNID<numNeighbors ; ++iNID){
      totalDamage += bondDamageNP1[bondIndex++];
    }
    if(numNeighbors > 0)
      totalDamage /= numNeighbors;
    else
      totalDamage = 0.0;
    damage[nodeId] = totalDamage;
  }
}
