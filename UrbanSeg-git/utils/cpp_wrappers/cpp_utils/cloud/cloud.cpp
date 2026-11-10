//
//
//		0==========================0
//		|    Local feature test    |
//		0==========================0
//
//		version 1.0 : 
//			> 
//
//---------------------------------------------------
//
//		Cloud source :
//		Define usefull Functions/Methods
//
//----------------------------------------------------
//
//		Hugues THOMAS - 10/02/2017
//


#include "cloud.h"


// Getters
// *******

 PointXYZ max_point(const PointXYZ A, const PointXYZ B)
{
	PointXYZ maxP;
	if (A.x > B.x) maxP.x = A.x;
	else maxP.x = B.x;

	if (A.y > B.y) maxP.y = A.y;
	else maxP.y = B.y;

	if (A.z > B.z) maxP.z = A.z;
	else maxP.z = B.z;

	return maxP;
}
PointXYZ min_point(const PointXYZ A, const PointXYZ B)
{
	PointXYZ minP;
	if (A.x < B.x) minP.x = A.x;
	else minP.x = B.x;

	if (A.y < B.y) minP.y = A.y;
	else minP.y = B.y;

	if (A.z < B.z) minP.z = A.z;
	else minP.z = B.z;

	return minP;
}

PointXYZ max_point(std::vector<PointXYZ> points)
{
	// Initiate limits
	PointXYZ maxP(points[0]);

	// Loop over all points
	for (auto p : points)
	{
		maxP = max_point(p, maxP);
	}

	return maxP;
}

PointXYZ min_point(std::vector<PointXYZ> points)
{
	// Initiate limits
	PointXYZ minP(points[0]);

	// Loop over all points
	for (auto p : points)
	{
		minP = min_point(p, minP);
	}

	return minP;
}