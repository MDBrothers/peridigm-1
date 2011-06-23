/*
 * Sortable.cxx
 *
 *  Created on: Mar 15, 2011
 *      Author: jamitch
 */
#include "Sortable.h"
#include "Array.h"
#include <algorithm>

namespace UTILITIES {
/*
 * THIS IS A PRIVATE FUNCTION
 */
Array<int> getPointsInNeighborhoodOfAxisAlignedMinimumValue
(
		CartesianComponent axis,
		double axisMinimumValue,
		double horizon,
		const Sortable& c,
		Array<int>& sortedMap
);

/*
 * THIS IS A PRIVATE FUNCTION
 */
Array<int> getPointsInNeighborhoodOfAxisAlignedMaximumValue
(
		CartesianComponent axis,
		double axisMaximumValue,
		double horizon,
		const Sortable& c,
		Array<int>& sortedMap
);

Array<int> getSortedMap(const Sortable& c, CartesianComponent axis){
	size_t numPoints = c.get_num_points();
	Sortable::Comparator compare = c.getComparator(axis);
	Array<int> sortedMap = c.getIdentityMap();
	/*
	 * Sort points
	 */
	std::sort(sortedMap.get(),sortedMap.get()+numPoints,compare);
	return sortedMap;
}

Array<int> getPointsInNeighborhoodOfAxisAlignedMinimumValue
(
		CartesianComponent axis,
		shared_ptr<double> xPtr,
		size_t numPoints,
		double horizon,
		double axisMinimumValue
)
{
	const Sortable c(numPoints,xPtr);
	Array<int> sortedMap = getSortedMap(c,axis);
	return getPointsInNeighborhoodOfAxisAlignedMinimumValue(axis,axisMinimumValue,horizon,c,sortedMap);
}

Array<int> getPointsInNeighborhoodOfAxisAlignedMinimumValue
(
		CartesianComponent axis,
		double axisMinimumValue,
		double horizon,
		const Sortable& c,
		Array<int>& sortedMap
){
	int numPoints = c.get_num_points();
	/*
	 * Look for points from "xMin" to "xMin+horizon"
	 */
	double value = axisMinimumValue + horizon;
	const Sortable::SearchIterator start=c.begin(axis,sortedMap.get_shared_ptr());
	const Sortable::SearchIterator end=start+numPoints;
	Sortable::SearchIterator upper = std::upper_bound(start,end,value);
	int numPointsInSet = upper.numPointsFromStart();
	Array<int> pointIds(numPointsInSet);
	int *ids = pointIds.get();
	for(const int *i=upper.mapStart();i!=upper.mapIterator();i++,ids++)
		*ids = *i;

	return pointIds;
}


/**
 * Mostly for rectangular type meshes; Returns points in plane perpendicular to "axis";
 * All points within a distance of horizon of the axis minimum are returned
 * @param axis -- X || Y || Z
 * @param horizon
 */
Array<int> getPointsAxisAlignedMinimum
(
		CartesianComponent axis,
		shared_ptr<double> xPtr,
		size_t numPoints,
		double horizon
)
{

	const Sortable c(numPoints,xPtr);
	Array<int> sortedMap = getSortedMap(c,axis);

	/*
	 * Get sorted points
	 */
	int *mapX = sortedMap.get();
	/*
	 * First value in map corresponds with minimum value of coordinate "axis"
	 */
	int iMIN = *mapX;
	double xMin = *(xPtr.get()+3*iMIN+axis);

	return getPointsInNeighborhoodOfAxisAlignedMinimumValue(axis,xMin,horizon,c,sortedMap);
}


Array<int> getPointsAxisAlignedMaximum
(
		CartesianComponent axis,
		shared_ptr<double> xPtr,
		size_t numPoints,
		double horizon
)
{

	const Sortable c(numPoints,xPtr);

	Array<int> sortedMap = getSortedMap(c,axis);
	/*
	 * Get sorted points
	 */
	int *mapX = sortedMap.get();

	/*
	 * Last value in map corresponds with maximum value
	 */
	int iMAX = *(mapX+numPoints-1);
	double max = *(xPtr.get()+3*iMAX+axis);

	return getPointsInNeighborhoodOfAxisAlignedMaximumValue(axis,max,horizon,c,sortedMap);
}

Array<int> getPointsInNeighborhoodOfAxisAlignedMaximumValue
(
		CartesianComponent axis,
		shared_ptr<double> xPtr,
		size_t numPoints,
		double horizon,
		double axisMaximumValue
)
{
	const Sortable c(numPoints,xPtr);
	Array<int> sortedMap = getSortedMap(c,axis);
	return getPointsInNeighborhoodOfAxisAlignedMaximumValue(axis,axisMaximumValue,horizon,c,sortedMap);
}

Array<int> getPointsInNeighborhoodOfAxisAlignedMaximumValue
(
		CartesianComponent axis,
		double axisMaximumValue,
		double horizon,
		const Sortable& c,
		Array<int>& sortedMap
){

	int numPoints = c.get_num_points();
	/*
	 * Look for points from "max-horizon" to "max"
	 */
	double value = axisMaximumValue - horizon;
	const Sortable::SearchIterator start=c.begin(axis,sortedMap.get_shared_ptr());
	const Sortable::SearchIterator end=start+numPoints;
	Sortable::SearchIterator upper = std::upper_bound(start,end,value);
	int numPointsInSet = upper.numPointsToEnd();
	Array<int> pointIds(numPointsInSet);
	int *ids = pointIds.get();
	for(const int *i=upper.mapIterator();i!=upper.mapEnd();i++,ids++)
		*ids = *i;

	return pointIds;

}

shared_ptr< std::set<int> > constructFrameSet(size_t num_owned_points, shared_ptr<double> owned_x, double horizon) {

	Sortable points(num_owned_points, owned_x);

	size_t numAxes=3;
	Array<CartesianComponent> components(numAxes);
	components[0] = UTILITIES::X;
	components[1] = UTILITIES::Y;
	components[2] = UTILITIES::Z;

	Array< Array<int> > sorted_maps(numAxes);
	for(size_t c=0;c<components.get_size();c++){

		Sortable::Comparator compare = points.getComparator(components[c]);
		sorted_maps[c] = points.getIdentityMap();
		/*
		 * Sort points
		 */
		std::sort(sorted_maps[c].get(),sorted_maps[c].get()+num_owned_points,compare);

	}

	/*
	 * Loop over axes and collect points at min and max ranges
	 * Add Points to frame set
	 */
	shared_ptr< std::set<int> >  frameSetPtr(new std::set<int>);
	for(size_t c=0;c<components.get_size();c++){

		CartesianComponent comp = components[c];

		Array<int> map_component = sorted_maps[c];

		{
			/*
			 * MINIMUM
			 * Find least upper bound of points for Min+horizon
			 */
			const double *x = points.get();
			int *map = map_component.get();
			/*
			 * First value in map corresponds with minimum value
			 */
			int iMIN = *map;
			double min = x[3*iMIN+comp];
			double value = min + horizon;
			const Sortable::SearchIterator start = points.begin(comp,map_component.get_shared_ptr());
			const Sortable::SearchIterator end = start+num_owned_points;
			Sortable::SearchIterator lub = std::upper_bound(start,end,value);
			frameSetPtr->insert(lub.mapStart(),lub.mapIterator());

		}

		{
			/*
			 * MAXIMUM
			 * Find greatest lower bound glb for Max-horizon
			 */
			const double *x = points.get();
			int *map = map_component.get();

			/*
			 * Last value in map corresponds with maximum value
			 */
			int iMAX = *(map+num_owned_points-1);
			double max = x[3*iMAX+comp];
			double value = max - horizon;
			const Sortable::SearchIterator start=points.begin(comp,map_component.get_shared_ptr());
			const Sortable::SearchIterator end=start+num_owned_points;
			Sortable::SearchIterator glb = std::upper_bound(start,end,value);
			frameSetPtr->insert(glb.mapIterator(),glb.mapEnd());
		}
	}

	return frameSetPtr;

}


} /* UTITILITIES */
