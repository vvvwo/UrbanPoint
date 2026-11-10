#include "../../cpp_utils/cloud/cloud.h"
#include <cmath>
#include <algorithm>
//#include "cloud.h"
using namespace std;

class VoxelData
{
public:
	vector<size_t> point_indexs;
	float max_height = 0;

	VoxelData()
	{
		point_indexs = vector<size_t>();
	}

	inline void AddPoint(size_t index, float height)
	{
		point_indexs.push_back(index);
		max_height = max(height, max_height);
	}
};

void calculation(vector<PointXYZ>& original_points, vector<float>& voxel_heights, float pillar_size);
