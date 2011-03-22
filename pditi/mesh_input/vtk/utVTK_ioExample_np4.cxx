/*
 * utVTK_ioExample.cxx
 *
 *  Created on: Oct 4, 2010
 *      Author: jamitch
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>
#include <tr1/memory>
#include "quick_grid/QuickGrid.h"
#include "PdutMpiFixture.h"
#include "vtk/PdVTK.h"
#include "vtk/Field.h"
#include "PdZoltan.h"

#include "Epetra_ConfigDefs.h"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include <iostream>


using namespace Field_NS;
using namespace Pdut;
using std::tr1::shared_ptr;
using namespace boost::unit_test;
using std::cout;

static size_t myRank;
static size_t numProcs;
const size_t nx = 10;
const size_t ny = 10;
const size_t nz = 1;
const double xStart = 0.0;
const double xLength = 1.0;
const double yStart = 0.0;
const double yLength = 1.0;
const double zStart = -0.5;
const double zLength = 1.0;
const QUICKGRID::Spec1D xSpec(nx,xStart,xLength);
const QUICKGRID::Spec1D ySpec(ny,yStart,yLength);
const QUICKGRID::Spec1D zSpec(nz,zStart,zLength);
const size_t numCells = nx*ny*nz;

Field<double> getPureShearXY(double gamma, const Field<double>& X, Field<double>& U){
	std::size_t numPoints = X.get_num_points();
	double *u = U.get();
	const double *x = X.get();

	for(std::size_t i=0;i<numPoints;i++){
		int p=3*i;
		u[p]=gamma*x[p+1];
		u[p+1]=0;
		u[p+2]=0;
	}
	return U;
}

QUICKGRID::QuickGridData getGrid() {

	double dx = xSpec.getCellSize();
	double dy = ySpec.getCellSize();
	double horizon = sqrt(dx*dx+dy*dy);
	QUICKGRID::TensorProduct3DMeshGenerator cellPerProcIter(numProcs,horizon,xSpec,ySpec,zSpec);
	QUICKGRID::QuickGridData decomp =  QUICKGRID::getDiscretization(myRank, cellPerProcIter);

	// This reload balances
	decomp = PDNEIGH::getLoadBalancedDiscretization(decomp);
	return decomp;
}



void utVTK_ioExample()
{
	QUICKGRID::QuickGridData pdGridData = getGrid();
	int numPoints = pdGridData.numPoints;

	/*
	 * NOTE
	 *
	 * FieldSpec is IMMUTABLE
	 *
	 */

	/*
	 * Create Spec(s) From Scratch
	 */
	const FieldSpec myRankSpec(FieldSpec::DEFAULT_FIELDTYPE,FieldSpec::SCALAR,"MyRank");
	const FieldSpec displacementSpec(FieldSpec::DISPLACEMENT,FieldSpec::VECTOR3D, "Displacement");
	const FieldSpec velocitySpec(FieldSpec::VELOCITY,FieldSpec::VECTOR3D, "v");
	const FieldSpec accelerationSpec(FieldSpec::ACCELERATION,FieldSpec::VECTOR3D, "a");

	/*
	 * Use existing spec (these are equivalent to the above -- but different 'names' on output
	 */
	const FieldSpec uSpec(DISPL3D);
	const FieldSpec vSpec(VELOC3D);
	const FieldSpec aSpec(ACCEL3D);

	const FieldSpec volSpec(VOLUME);
	const FieldSpec wSpec(WEIGHTED_VOLUME);
	const FieldSpec thetaSpec(DILATATION);

	/*
	 * This is not required in Peridigm
	 */
	Field<double> X(COORD3D,pdGridData.myX,numPoints);
	Field<double> uField(uSpec,numPoints), vField(vSpec,numPoints),aField(aSpec,numPoints);
	Field<double> wField(wSpec,numPoints), thetaField(thetaSpec,numPoints);
	Field<int> rankField(myRankSpec,numPoints);
	uField.set(0.0); vField.set(0.0); aField.set(0.0);
	wField.set(0.0); thetaField.set(0.0);
	rankField.set(myRank);

	/*
	 * RAW POINTERS; GET THESE from Epetra_Vector
	 */
	const double *xPtr = X.get();
	double *uPtr = uField.get();
	double *vPtr = vField.get();
	double *aPtr = aField.get();
	double *volPtr = pdGridData.cellVolume.get();
	double *wPtr = wField.get();
	double *thetaPtr = thetaField.get();
	const int *neighPtr = pdGridData.neighborhood.get();

	/*
	 * Create VTK unstructured grid
	 */
	vtkSmartPointer<vtkUnstructuredGrid> grid = PdVTK::getGrid(pdGridData.myX,numPoints);

	/*
	 * Create example 'collection' writers
	 * 1) ascii
	 * 2) binary
	 */
	PdVTK::CollectionWriter asciiWriter("utVTK_ioExample_ascii",numProcs, myRank, PdVTK::vtkASCII);
	PdVTK::CollectionWriter binWriter("utVTK_ioExample_bin",numProcs, myRank, PdVTK::vtkBINARY);

	/*
	 * Write fields
	 * This doesn't actually write until later; Here it just sets the pointers
	 */
	PdVTK::writeField<double>(grid,uSpec,uPtr);
	PdVTK::writeField(grid,vSpec,vPtr);
	PdVTK::writeField(grid,aSpec,aPtr);
	PdVTK::writeField(grid,volSpec,volPtr);
	PdVTK::writeField(grid,wSpec,wPtr);
	PdVTK::writeField(grid,thetaSpec,thetaPtr);
	PdVTK::writeField(grid,rankField);

	/*
	 * Example that loops over time
	 */

	std::size_t numSteps = 10;
	const QUICKGRID::Spec1D gammaSpec(numSteps,0,.1);
	double dGamma=gammaSpec.getCellSize();

	/*
	 * Write initial conditions: gamma=0
	 */
	double gamma=0.0;
	asciiWriter.writeTimeStep(gamma,grid);
	binWriter.writeTimeStep(gamma,grid);
	for(std::size_t j=0;j<numSteps;j++){

		/*
		 * Do time integration and physics
		 */
		gamma += dGamma;
		getPureShearXY(gamma, X, uField);


		asciiWriter.writeTimeStep(gamma,grid);
		binWriter.writeTimeStep(gamma,grid);


	}

	/*
	 * This writes the "pvd" collection file
	 */
	asciiWriter.close("mpiexec -np 4 ./utVTK_ioExample_np4\n");
	binWriter.close();



}



bool init_unit_test_suite()
{
	// Add a suite for each processor in the test
	bool success=true;
	test_suite* proc = BOOST_TEST_SUITE( "utVTK_ioExample" );
	proc->add(BOOST_TEST_CASE( &utVTK_ioExample ));
	framework::master_test_suite().add( proc );
	return success;
}


bool init_unit_test()
{
	init_unit_test_suite();
	return true;
}

int main
(
		int argc,
		char* argv[]
)
{

	// Initialize MPI and timer
	PdutMpiFixture myMpi = PdutMpiFixture(argc,argv);

	// These are static (file scope) variables
	myRank = myMpi.rank;
	numProcs = myMpi.numProcs;

	/**
	 * This test only make sense for numProcs == 4
	 */
	if(4 != numProcs){
		std::cerr << "Unit test runtime ERROR: utVTK_ioExample_np4 only makes sense on 4 processors" << std::endl;
		std::cerr << "\t Re-run unit test $mpiexec -np 4 ./utVTK_ioExample_np4" << std::endl;
		myMpi.PdutMpiFixture::~PdutMpiFixture();
		std::exit(-1);
	}

	// Initialize UTF
	return unit_test_main( init_unit_test, argc, argv );
}

