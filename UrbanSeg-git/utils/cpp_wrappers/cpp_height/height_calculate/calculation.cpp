#include "calculation.h"

void calculation(vector<PointXYZ>& original_points, vector<float>& voxel_heights,float pillar_size)
{
	size_t N = original_points.size();
	PointXYZ minCorner = min_point(original_points);
	PointXYZ maxCorner = max_point(original_points);


	size_t sampleNX = (size_t)floor((maxCorner.x - minCorner.x) / pillar_size) + 1;
	size_t sampleNY = (size_t)floor((maxCorner.y - minCorner.y) / pillar_size) + 1;

	unordered_map<size_t, VoxelData> data;
	size_t i = 0;
	size_t iX, iY, voxel_idx;
	voxel_heights.resize(N);

	for (auto& p : original_points)
	{
		iX = (size_t)floor((p.x - minCorner.x) / pillar_size);
		iY = (size_t)floor((p.y - minCorner.y) / pillar_size);

		if (iX > sampleNX - 1) iX--;
		if (iY > sampleNY - 1) iY--;

		voxel_idx = iX + sampleNX * iY;

		if (data.count(voxel_idx) < 1)
			data.emplace(voxel_idx, VoxelData());

		data[voxel_idx].AddPoint(i, p.z);

		i++;
	}

	for (auto& item : data)
	{
		for (auto& id : item.second.point_indexs)
		{
			voxel_heights[id] = item.second.max_height;
		}
	}
}