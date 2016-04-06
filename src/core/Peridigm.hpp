/*! \file Peridigm.hpp */

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


#ifndef PERIDIGM_HPP
#define PERIDIGM_HPP

#include <vector>
#include <set>

#include <BelosLinearProblem.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosEpetraAdapter.hpp>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <NOX.H>
#include <NOX_Epetra.H>
#include <NOX_Epetra_Interface_Required.H>
#include <NOX_Epetra_Interface_Jacobian.H>
#include <NOX_Epetra_Interface_Preconditioner.H>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include "Peridigm_Block.hpp"
#include "Peridigm_Discretization.hpp"
#include "Peridigm_ModelEvaluator.hpp"
#include "Peridigm_DataManager.hpp"
#include "Peridigm_SerialMatrix.hpp"
#include "Peridigm_OutputManagerContainer.hpp"
#include "Peridigm_ComputeManager.hpp"
#include "Peridigm_BoundaryAndInitialConditionManager.hpp"
#include "Peridigm_ContactManager.hpp"
#include "Peridigm_ServiceManager.hpp"
#include "Peridigm_Memstat.hpp"
#include "Peridigm_Material.hpp"
#include "Peridigm_DamageModel.hpp"

namespace PeridigmNS {

  class UserDefinedTimeDependentCriticalStretchDamageModel;

  class Peridigm : public NOX::Epetra::Interface::Required, public NOX::Epetra::Interface::Jacobian, public NOX::Epetra::Interface::Preconditioner {

  public:

    //! Constructor
    Peridigm(const MPI_Comm& peridigmComm,
             Teuchos::RCP<Teuchos::ParameterList> params,
             Teuchos::RCP<Discretization> inputPeridigmDiscretization);

    //! Destructor
    ~Peridigm(){};

    //! Return a string containing the version number
    std::string version(){ return std::string("1.0.0"); }

    //! Initialize discretization and maps
    void initializeDiscretization(Teuchos::RCP<Discretization> peridigmDisc);

    //! Throws a warning if it seems like the contact search radius is too big
    void checkContactSearchRadius(const Teuchos::ParameterList& contactParams, Teuchos::RCP<Discretization> peridigmDisc);

    //! Initialize the workset
    void initializeWorkset();

   //! Initialize Restart
    void InitializeRestart();

    //! Instantiate the compute manager
    void instantiateComputeManager(Teuchos::RCP<Discretization> peridigmDiscretization);

    //! Initialize the output manager
    void initializeOutputManager();

    //! Initialize blocks
    void initializeBlocks(Teuchos::RCP<Discretization> disc);

    //! Main routine to drive time integration
    void execute(Teuchos::RCP<Teuchos::ParameterList> solverParams);

    //! Called from Main to drive multiple time integration solvers in sequence
    void executeSolvers();

    //! Compute the residual vector (pure virtual method in NOX::Epetra::Interface::Required)
    bool computeF(const Epetra_Vector& x, Epetra_Vector& FVec, FillType fillType = Residual);

    //! Compute the Jacobian (pure virtual method in NOX::Epetra::Interface::Jacobian)
    bool computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac);

    //! Current age of the 3x3 block preconditioner for NOX solves
    int agePeridigmPreconditioner;

    //! Maximum allowable age of the 3x3 block preconditioner for NOX solves
    int maxAgePeridigmPreconditioner;

    //! Number of nonlinear iterations between Jacobian updates for NOX Nonlinear CG solves.
    int m_noxTriggerJacobianUpdate;

    //! Counter for updating the Jacobian for NOX Nonlinear CG solves.
    int m_noxJacobianUpdateCounter;

    //! Compute the preconditioner (pure virtual method in NOX::Epetra::Interface::Preconditioner)
    bool computePreconditioner(const Epetra_Vector& x, Epetra_Operator& M, Teuchos::ParameterList* precParams = 0);

    //! Residual and Jacobian matrix fills for NOX interface
    virtual bool evaluateNOX(FillType f, const Epetra_Vector *solnVector, Epetra_Vector *rhsVector);

    //! Returns true if tangent has been allocated, false otherwise.
    bool hasTangentStiffnessMatrix(){
      bool hasTangent = false;
      if(!tangent.is_null()){
	hasTangent = true;
      }
      return hasTangent;
    }

    //! Returns the global tangent stiffness matrix (intended for use when calling Peridigm as a library).
    Teuchos::RCP<const Epetra_FECrsMatrix> getTangentStiffnessMatrix(){
      TEUCHOS_TEST_FOR_EXCEPT_MSG(tangent.is_null(), "**** PeridigmNS::Peridigm::getTangentStiffnessMatrix(), tangent has not been allocated!\n");
      return tangent;
    }

    //! Evaluate the tangent stiffness matrix (intended for use when calling Peridigm as a library).
    void evaluateTangentStiffnessMatrix(){
      TEUCHOS_TEST_FOR_EXCEPT_MSG(tangent.is_null(), "**** PeridigmNS::Peridigm::evaluateTangentStiffnessMatrix(), tangent has not been allocated!\n");
      tangent->PutScalar(0.0);
      modelEvaluator->evalJacobian(workset);
      int err = tangent->GlobalAssemble();
      TEUCHOS_TEST_FOR_EXCEPT_MSG(err != 0, "**** PeridigmNS::Peridigm::evaluateTangentStiffnessMatrix(), GlobalAssemble() returned nonzero error code.\n");
      // Note:  Peridigm expects the tangent to be scaled using tangent->Scale(-1.0);
      //        but Albany does not.
    }

    //! Evaluate the internal force; assumes x, u, y, and v have been set, fills force (intended for use when calling Peridigm as a library).
    void computeInternalForce();

    // Update the material states (intended for use when calling Peridigm as a library).
    void updateState() {
      for(blockIt = blocks->begin() ; blockIt != blocks->end() ; blockIt++)
        blockIt->updateState();
    }

    // Write the Peridigm sub-model to an Exodus file (intended for use when calling Peridigm as a library).
    void writePeridigmSubModel(double currentTime) {
      synchDataManagers();
      outputManager->write(blocks, currentTime);
    }

    //! Get the field manager (intended for use when calling Peridigm as a library).
    Teuchos::RCP<PeridigmNS::FieldManager> getFieldManager() {
      Teuchos::RCP<PeridigmNS::FieldManager> fieldManager = Teuchos::rcpFromRef(PeridigmNS::FieldManager::self());
      return fieldManager;
    }

    //! Query the presence of a specific data field in a given block's DataManger (intended for use when calling Peridigm as a library).
    bool hasBlockData(std::string blockName, std::string fieldName) {

      // locate the block by name
      std::vector<PeridigmNS::Block>::iterator block;
      bool found = false;
      for(blockIt = blocks->begin() ; blockIt != blocks->end() ; blockIt++){
        if(blockIt->getName() == blockName){
          block = blockIt;
          found = true;
          continue;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPT_MSG(!found, "**** Error, Peridigm::getBlockData(), block " + blockName + " not found!\n");

      PeridigmNS::FieldManager& fieldManager = PeridigmNS::FieldManager::self();

      if(!fieldManager.hasField(fieldName)){
	return false;
      }

      int fieldId = fieldManager.getFieldId(fieldName);
      PeridigmNS::FieldSpec spec = fieldManager.getFieldSpec(fieldId);
      PeridigmNS::PeridigmField::Step step = PeridigmNS::PeridigmField::STEP_NONE;
      if(spec.getTemporal() == PeridigmNS::PeridigmField::TWO_STEP)
        step = PeridigmNS::PeridigmField::STEP_NP1;

      return block->hasData(fieldId, step);
    }

    //! Get an Epetra_Vector of data from a given block's DataManger (intended for use when calling Peridigm as a library).
    Teuchos::RCP<Epetra_Vector> getBlockData(std::string blockName, std::string fieldName) {

      // \todo Organize all the functions that are meant to be called in library mode.

      // \todo Call synchDataManagers() first?

      // \todo store this mapping in a map to avoid the repeated search.
      std::vector<PeridigmNS::Block>::iterator block;
      bool found = false;
      for(blockIt = blocks->begin() ; blockIt != blocks->end() ; blockIt++){
        if(blockIt->getName() == blockName){
          block = blockIt;
          found = true;
          continue;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPT_MSG(!found, "**** Error, Peridigm::getBlockData(), block " + blockName + " not found!\n");

      // \todo Keep a map of field name to field id in Albany and avoid the loop below.
      PeridigmNS::FieldManager& fieldManager = PeridigmNS::FieldManager::self();
      int fieldId = fieldManager.getFieldId(fieldName);
      PeridigmNS::FieldSpec spec = fieldManager.getFieldSpec(fieldId);
      PeridigmNS::PeridigmField::Step step = PeridigmNS::PeridigmField::STEP_NONE;
      if(spec.getTemporal() == PeridigmNS::PeridigmField::TWO_STEP)
        step = PeridigmNS::PeridigmField::STEP_NP1;
      Teuchos::RCP<Epetra_Vector> data = block->getData(fieldId, step);
      return data;
    }

    //! Perform diagnostics on Jacobian and print results to screen.
    void jacobianDiagnostics(Teuchos::RCP<NOX::Epetra::Group> noxGroup);

    void executeExplicit(Teuchos::RCP<Teuchos::ParameterList> solverParams);

    //! Main routine to drive problem solution for quasistatics
    void executeQuasiStatic(Teuchos::RCP<Teuchos::ParameterList> solverParams);

    //! Main routine to drive problem solution for quasistatics using NOX
    void executeNOXQuasiStatic(Teuchos::RCP<Teuchos::ParameterList> solverParams);

    //! Set the preconditioner for the global linear system
    void quasiStaticsSetPreconditioner(Belos::LinearProblem<double,Epetra_MultiVector,Epetra_Operator>& linearProblem);

    //! Damp the tangent matrix by scaling the diagonal and adding a small value to each entry in the diagonal
    void quasiStaticsDampTangent(double dampedNewtonDiagonalScaleFactor,
                                 double dampedNewtonDiagonalShiftFactor);

    //! Solve the global linear system
    Belos::ReturnType quasiStaticsSolveSystem(Teuchos::RCP<Epetra_Vector> residual,
                                              Teuchos::RCP<Epetra_Vector> lhs,
                                              Belos::LinearProblem<double,Epetra_MultiVector,Epetra_Operator>& linearProblem,
                                              Teuchos::RCP< Belos::SolverManager<double,Epetra_MultiVector,Epetra_Operator> >& belosSolver);

    //! Perform line search
    double quasiStaticsLineSearch(Teuchos::RCP<Epetra_Vector> residual,
                                  Teuchos::RCP<Epetra_Vector> lhs,
                                  double dt);

    //! Main routine to drive problem solution with implicit time integration
    void executeImplicit(Teuchos::RCP<Teuchos::ParameterList> solverParams);

    //! Allocate memory for non-zeros in global Jacobian
    void allocateJacobian(const int numDoFs);

    //! Allocate memory for non-zeros in block diagonal Jacobian
    void allocateBlockDiagonalJacobian();

    //! Compute the Jacobian for implicit dynamics
    void computeImplicitJacobian(double beta, double dt);

    //! Compute the residual for quasi-statics
    double computeQuasiStaticResidual(Teuchos::RCP<Epetra_Vector> residual);

    //! Synchronize data in DataManagers across processes (needed before call to OutputManager::write() )
    void synchDataManagers();

    //! Accessor for comm object
    Teuchos::RCP<const Epetra_Comm> getEpetraComm(){ return peridigmComm; }

    //! @name Accessors for maps
    //@{
    Teuchos::RCP<const Epetra_BlockMap> getOneDimensionalMap() { return oneDimensionalMap; }
    Teuchos::RCP<const Epetra_BlockMap> getThreeDimensionalMap() { return threeDimensionalMap; }
    Teuchos::RCP<const Epetra_BlockMap> getBondMap() { return bondMap; }
    Teuchos::RCP<const Epetra_BlockMap> getOneDimensionalOverlapMap() { return oneDimensionalOverlapMap; }
    //@}

    //! @name Accessors for main solver-level vectors
    //@{
    Teuchos::RCP<Epetra_Vector> getBlockIDs() { return blockIDs; }
    Teuchos::RCP<Epetra_Vector> getX() { return x; }
    Teuchos::RCP<Epetra_Vector> getU() { return u; }
    Teuchos::RCP<Epetra_Vector> getY() { return y; }
    Teuchos::RCP<Epetra_Vector> getV() { return v; }
    Teuchos::RCP<Epetra_Vector> getA() { return a; }
    Teuchos::RCP<Epetra_Vector> getMatrixPorosity() {return matrixPorosity; }
    Teuchos::RCP<Epetra_Vector> getFractureProrosity() {return fracturePorosity; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneSaturationPoresY() { return phaseOneSaturationPoresY; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneSaturationFracY() { return phaseOneSaturationFracY; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneSaturationPoresU() { return phaseOneSaturationPoresU; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneSaturationFracU() { return phaseOneSaturationFracU; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneSaturationPoresDeltaU() { return phaseOneSaturationPoresIncrement; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneSaturationFracDeltaU() { return phaseOneSaturationFracIncrement; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneSaturationPoresV() { return phaseOneSaturationPoresV; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneSaturationFracV() { return phaseOneSaturationFracV; }
    Teuchos::RCP<Epetra_Vector> getPorePressureU() { return porePressureU; }
    Teuchos::RCP<Epetra_Vector> getPorePressureY() { return porePressureY; }
    Teuchos::RCP<Epetra_Vector> getPorePressureV() { return porePressureV; }
    Teuchos::RCP<Epetra_Vector> getPorePressureDeltaU() { return porePressureDeltaU;}
    Teuchos::RCP<Epetra_Vector> getFracPressureU() { return fracturePressureU; }
    Teuchos::RCP<Epetra_Vector> getFracPressureY() { return fracturePressureY; }
    Teuchos::RCP<Epetra_Vector> getFracPressureV() { return fracturePressureV; }
    Teuchos::RCP<Epetra_Vector> getFracPressureDeltaU() { return fracturePressureDeltaU;}
    Teuchos::RCP<Epetra_Vector> getPhaseOnePoreFlow() { return phaseOnePoreFlow; }
    Teuchos::RCP<Epetra_Vector> getExternalPhaseOnePoreFlow() {return externalPhaseOnePoreFlow; }
    Teuchos::RCP<Epetra_Vector> getPhaseOneFracFlow() { return phaseOneFracFlow; }
    Teuchos::RCP<Epetra_Vector> getExternalPhaseOneFracFlow() {return externalPhaseOneFracFlow; }
    Teuchos::RCP<Epetra_Vector> getPhaseTwoPoreFlow() { return phaseTwoPoreFlow; }
    Teuchos::RCP<Epetra_Vector> getExternalPhaseTwoPoreFlow() {return externalPhaseTwoPoreFlow; }
    Teuchos::RCP<Epetra_Vector> getPhaseTwoFracFlow() { return phaseTwoFracFlow; }
    Teuchos::RCP<Epetra_Vector> getExternalPhaseTwoFracFlow() {return externalPhaseTwoFracFlow; }
    Teuchos::RCP<Epetra_Vector> getForce() { return force; }
    Teuchos::RCP<Epetra_Vector> getExternalForce() { return externalForce; }
    Teuchos::RCP<Epetra_Vector> getContactForce() { return contactForce; }
    Teuchos::RCP<Epetra_Vector> getDeltaU() { return deltaU; }
    Teuchos::RCP<Epetra_Vector> getVolume() { return volume; }
    Teuchos::RCP<Epetra_Vector> getDeltaTemperature() { return deltaTemperature; }
    //@}

    //! Accessor for global neighborhood data
    Teuchos::RCP<const PeridigmNS::NeighborhoodData> getGlobalNeighborhoodData() { return globalNeighborhoodData; }

    //! Accessor for interface data
    Teuchos::RCP<PeridigmNS::InterfaceData> getInterfaceData() { return interfaceData; }

    //! Flag if has interface data:
    bool interfacesAreConstructed(){return constructInterfaces;}

    //! Accessor for vector of Blocks
    Teuchos::RCP< std::vector<PeridigmNS::Block> > getBlocks() { return blocks; }

    //! Accessor for individual Blocks
    Teuchos::RCP<PeridigmNS::Block> getBlock(unsigned int blockNumber) {
      return Teuchos::rcpFromRef(blocks->at(blockNumber));
    }

    //! Accessor for node sets
    Teuchos::RCP< std::map< std::string, std::vector<int> > > getExodusNodeSets();

    //! Accessor for compute manager
    Teuchos::RCP< PeridigmNS::ComputeManager > getComputeManager() { return computeManager; }

    //! Set the time step (for use when calling Peridigm as a library).
    void setTimeStep(double timeStep) { workset->timeStep = timeStep; }

    //! Display a progress bar
    void displayProgress(std::string title, double percentComplete);

    //! Display information about memory usage
    void printMemoryStats(){Memstat * memstat = Memstat::Instance(); memstat->printStats();};

  private:

    //! @name Friend classes
    //@{
    friend class OutputManager_ExodusII;
    //@}

    //! Parameterlist of entire input deck
    Teuchos::RCP<Teuchos::ParameterList> peridigmParams;

    //! Epetra communicator established by Peridigm_Factory
    Teuchos::RCP<const Epetra_Comm> peridigmComm;

    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! Maps for scalar, vector, and bond data
    Teuchos::RCP<const Epetra_BlockMap> oneDimensionalMap;
    Teuchos::RCP<const Epetra_BlockMap> threeDimensionalMap;
    Teuchos::RCP<const Epetra_BlockMap> nDimensionalMap;
    Teuchos::RCP<const Epetra_BlockMap> bondMap;
    Teuchos::RCP<const Epetra_BlockMap> oneDimensionalOverlapMap;

    //! Global current time
    double currentTime;

    //! Contact flag
    bool analysisHasContact;

    //! Multiphysics flag
    bool twoPhasePoroelasticity;
    bool onePhasePoroelasticity;

    //! Flag for computing element-sphere intersections
    bool computeIntersections;

    //! Flag for computing interface information
    bool constructInterfaces;

    //! Damage models
    std::map< std::string, Teuchos::RCP<const PeridigmNS::DamageModel> > damageModels;

    Teuchos::RCP< PeridigmNS::UserDefinedTimeDependentCriticalStretchDamageModel > CSDamageModel;

    //! Boundary and initial condition manager
    Teuchos::RCP<PeridigmNS::BoundaryAndInitialConditionManager> boundaryAndInitialConditionManager;

    //! Contact manager
    Teuchos::RCP<PeridigmNS::ContactManager> contactManager;

    //! Compute manager
    Teuchos::RCP<PeridigmNS::ComputeManager> computeManager;

    //! Parameterlist containing global data (data not stored in a data manager) to a compute class
    Teuchos::RCP<Teuchos::ParameterList> computeClassGlobalData;

    //! Service manager
    Teuchos::RCP<PeridigmNS::ServiceManager> serviceManager;

    //! Mothership multivector that contains all the three-dimensional global vectors (x, u, y, v, a, force, etc.)
    Teuchos::RCP<Epetra_MultiVector> threeDimensionalMothership;

    //! Mothership multivector that contains all the n-dimensional global vectors (x, u, y, v, a, force, etc.)
    Teuchos::RCP<Epetra_MultiVector> nDimensionalMothership;

    //! Mothership multivector that contains all the one-dimensional global vectors (blockID, volume)
    Teuchos::RCP<Epetra_MultiVector> oneDimensionalMothership;

    //! Blocks
    Teuchos::RCP< std::vector<PeridigmNS::Block> > blocks;

    //! Block iterator, for convenience
    std::vector<PeridigmNS::Block>::iterator blockIt;

    //! Overlap Jacobian; filled by each processor and then assembled into the mothership Jacobian;
    Teuchos::RCP<PeridigmNS::SerialMatrix> overlapJacobian;

    //! Global vector for initial positions
    Teuchos::RCP<Epetra_Vector> x;

    //! Global vector for displacement
    Teuchos::RCP<Epetra_Vector> u;

    //! Global vector for current position
    Teuchos::RCP<Epetra_Vector> y;

    //! Global vector for velocity
    Teuchos::RCP<Epetra_Vector> v;

    //! Global vector for acceleration
    Teuchos::RCP<Epetra_Vector> a;

    //! Global vector for temperature change
    Teuchos::RCP<Epetra_Vector> deltaTemperature;

    //! Global vector for force
    Teuchos::RCP<Epetra_Vector> force;

    //! Global vector for forces and flows
    Teuchos::RCP<Epetra_Vector> combinedForce;

    //! Global vector for contact force (used only in simulations with contact)
    Teuchos::RCP<Epetra_Vector> contactForce;

    //! Global vector for external forces
    Teuchos::RCP<Epetra_Vector> externalForce;

    //! Global vector for delta u (used only in implicit time integration)
    Teuchos::RCP<Epetra_Vector> deltaU;

    //! Global scratch space vector
    Teuchos::RCP<Epetra_Vector> scratch;
    Teuchos::RCP<Epetra_Vector> scratchOneD;

    //! Vector containing velocities at dof with kinematic bc; used only by NOX solver.
    Teuchos::RCP<Epetra_Vector> noxVelocityAtDOFWithKinematicBC;
    Teuchos::RCP<Epetra_Vector> noxPorePressureVAtDOFWithKinematicBC;
    Teuchos::RCP<Epetra_Vector> noxFracPressureVAtDOFWithKinematicBC;
    Teuchos::RCP<Epetra_Vector> noxPhaseOneSaturationPoresVAtDOFWithKinematicBC;
    Teuchos::RCP<Epetra_Vector> noxPhaseOneSaturationFracVAtDOFWithKinematicBC;

    //! Global vector for block ID
    Teuchos::RCP<Epetra_Vector> blockIDs;

    //! Global vector containing the horizon for each point
    Teuchos::RCP<Epetra_Vector> horizon;

    //! Global vector for cell volume
    Teuchos::RCP<Epetra_Vector> volume;

    //! Global vector for cell density
    Teuchos::RCP<Epetra_Vector> density;

    Teuchos::RCP<Epetra_Vector> porePressureIncrement;
    //! Global vector for pore pressure displacement analogue
    Teuchos::RCP<Epetra_Vector> porePressureU;
    //! Global vector for pore pressure current
    Teuchos::RCP<Epetra_Vector> porePressureY;
    //! Global vector for pore pressure current velocity analogue
    Teuchos::RCP<Epetra_Vector> porePressureV;
    //! Global vector for delta pore pressure
    Teuchos::RCP<Epetra_Vector> porePressureDeltaU;

    Teuchos::RCP<Epetra_Vector> fracturePressureIncrement; // needed to update spec two quantities
    //! Global vector for fracture pressure displacement analogue
    Teuchos::RCP<Epetra_Vector> fracturePressureU;
    //! Global vector for fracture pressure current
    Teuchos::RCP<Epetra_Vector> fracturePressureY;
    //! Global vector for fracture pressure current velocity analogue
    Teuchos::RCP<Epetra_Vector> fracturePressureV;
    //! Global vector for delta fracture pressure
    Teuchos::RCP<Epetra_Vector> fracturePressureDeltaU;

    //! Global vectors for matrix and fracture prorosities, the volume fraction available to fluid occupancy
    Teuchos::RCP<Epetra_Vector> matrixPorosity;
    Teuchos::RCP<Epetra_Vector> fracturePorosity;

    //! Global vector for phase one fraction in pores
    Teuchos::RCP<Epetra_Vector> phaseOneSaturationPoresIncrement;
    Teuchos::RCP<Epetra_Vector> phaseOneSaturationPoresV;
    Teuchos::RCP<Epetra_Vector> phaseOneSaturationPoresY;
    Teuchos::RCP<Epetra_Vector> phaseOneSaturationPoresU;

    //! Global vector for phase one fraction in fracture
    Teuchos::RCP<Epetra_Vector> phaseOneSaturationFracIncrement;
    Teuchos::RCP<Epetra_Vector> phaseOneSaturationFracV;
    Teuchos::RCP<Epetra_Vector> phaseOneSaturationFracY;
    Teuchos::RCP<Epetra_Vector> phaseOneSaturationFracU;

    //! Global vector for phaseOnePore flow
    Teuchos::RCP<Epetra_Vector> phaseOnePoreFlow;
    Teuchos::RCP<Epetra_Vector> externalPhaseOnePoreFlow;

    //! Global vector for phaseOneFrac flow
    Teuchos::RCP<Epetra_Vector> phaseOneFracFlow;
    //! Global vector for external phaseOneFrac flow inputs
    Teuchos::RCP<Epetra_Vector> externalPhaseOneFracFlow;

    //! Global vector for phaseTwoPore flow
    Teuchos::RCP<Epetra_Vector> phaseTwoPoreFlow;
    Teuchos::RCP<Epetra_Vector> externalPhaseTwoPoreFlow;

    //! Global vector for phaseTwoFrac flow
    Teuchos::RCP<Epetra_Vector> phaseTwoFracFlow;
    //! Global vector for external phaseTwoFrac flow inputs
    Teuchos::RCP<Epetra_Vector> externalPhaseTwoFracFlow;

    //! Type of tangent to evaluate
    PeridigmNS::Material::JacobianType jacobianType;

    //! Map for global tangent matrix (note, must be an Epetra_Map, not an Epetra_BlockMap)
    Teuchos::RCP<Epetra_Map> tangentMap;

    //! Map for block diagonal tangent matrix (note, must be an Epetra_Map, not an Epetra_BlockMap)
    Teuchos::RCP<Epetra_Map> blockDiagonalTangentMap;

    //! Global tangent matrix
    Teuchos::RCP<Epetra_FECrsMatrix> tangent;

    //! Block diagonal of global tangent matrix
    Teuchos::RCP<Epetra_FECrsMatrix> blockDiagonalTangent;

    //! Tracker for total number of iterations taken by the nonlinear solver for implicit time integration
    Teuchos::RCP<int> nonlinearSolverIterations;

    //! List of neighbors for all locally-owned nodes
    Teuchos::RCP<PeridigmNS::NeighborhoodData> globalNeighborhoodData;

    // information about internal interfaces in the domain
    Teuchos::RCP<PeridigmNS::InterfaceData> interfaceData;

    //! Workset that is passed to the modelEvaluator
    Teuchos::RCP<Workset> workset;

    //! The peridigm model evaluator
    Teuchos::RCP<PeridigmNS::ModelEvaluator> modelEvaluator;

    //! The peridigm output manager
    Teuchos::RCP<PeridigmNS::OutputManagerContainer> outputManager;

    //! Vector of parameters for each solver (multiple solvers indicates, e.g., a simulation with both implicit and explicit time integration)
    std::vector< Teuchos::RCP<Teuchos::ParameterList> > solverParameters;

    //! BLAS for local-only vector updates (BLAS-1)
    Epetra_BLAS blas;

    // field ids for all relevant data
    int elementIdFieldId;
    int blockIdFieldId;
    int horizonFieldId;
    int volumeFieldId;
    int modelCoordinatesFieldId;
    int coordinatesFieldId;
    int displacementFieldId;
    int velocityFieldId;
    int accelerationFieldId;
    int deltaTemperatureFieldId;
    int forceDensityFieldId;
    int contactForceDensityFieldId;
    int externalForceDensityFieldId;
    int partialVolumeFieldId;

    // multiphyics information
    int porePressureYFieldId;
    int porePressureUFieldId;
    int porePressureVFieldId;
    int fracturePressureYFieldId;
    int fracturePressureUFieldId;
    int fracturePressureVFieldId;

    int matrixPorosityFieldId;
    int fracturePorosityFieldId;

    int phaseOneSaturationPoresYFieldId;
    int phaseOneSaturationPoresUFieldId;
    int phaseOneSaturationPoresVFieldId;
    int phaseOneSaturationPoresIncrementFieldId;
    int phaseOneSaturationFracYFieldId;
    int phaseOneSaturationFracUFieldId;
    int phaseOneSaturationFracVFieldId;
    int phaseOneSaturationFracIncrementFieldId;

    int phaseOnePoreFlowDensityFieldId;
    int phaseOnePoreExternalFlowDensityFieldId;
    int phaseOneFracFlowDensityFieldId;
    int phaseOneFracExternalFlowDensityFieldId;

    int phaseTwoPoreFlowDensityFieldId;
    int phaseTwoPoreExternalFlowDensityFieldId;
    int phaseTwoFracFlowDensityFieldId;
    int phaseTwoFracExternalFlowDensityFieldId;

    int numMultiphysDoFs;
    string textMultiphysDoFs;

    // Map for restart files
    map<string, string> restartFiles;

    // Set name of restart files
    void setRestartNames(char const * path);

    // Write the restart files
    void writeRestart(Teuchos::RCP<Teuchos::ParameterList> solverParams);

    //Read the restart files
    void readRestart();
  };
}

#endif // PERIDIGM_HPP
