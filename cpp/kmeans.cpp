// kmeans.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <vector>
#include <cstdlib>
#include <set>
#include <map>
#include <cmath>
#include <iostream>
using namespace std;

struct Cluster {
	double x;
	double y;
	int clusterIndex;
	Cluster(double x, double y):x(x),y(y){}
};

struct Point {
	int x;
	int y;
	Point(int x, int y) :x(x), y(y) {

	}
};

vector<Point*> transform(vector<vector<int>> &points) {
	vector<Point*> newPoints;
	for (auto point : points) {
		Point *p = new Point(point[0], point[1]);
		newPoints.push_back(p);
	}
	return newPoints;
}

map<int, Cluster*> init(vector<Point*> &newPoints, int k) {
	map<int, Cluster*> kcluster;
	int n = newPoints.size();
	set<int> initIndexSet;
	while (true) {
		if (initIndexSet.size() == k) break;
		int index = rand() % n;
		if (initIndexSet.find(index) == initIndexSet.end()) {
			initIndexSet.insert(index);
		}
	}
	int i = 0;
	for (int index : initIndexSet) {
		Cluster *cluster = new Cluster(newPoints[index]->x, newPoints[index]->y);
		cluster->clusterIndex = ++i;
		kcluster[i]=cluster;
	}
	return kcluster;
}

double computeDis(Point *p1, Cluster* p2) {
	return pow((p1->x - p2->x), 2) + pow((p1->y - p2->y), 2);
}
void updateCordinate(Cluster* cluster, vector<Point*> &point) {
	int n = point.size();
	double x = 0;
	double y = 0;
	for (Point* p : point) {
		x += p->x;
		y += p->y;
	}
	cluster->x = x / n;
	cluster->y = y / n;
}

void updateCluster(map<int, vector<Point*>> &cluster2point, map<int, Cluster*> &index2kclusters){
	for (auto kv : cluster2point) {
		int index = kv.first;
		vector<Point*> points = kv.second;//TODO
		Cluster *cluster = index2kclusters[index];
		updateCordinate(cluster, points);
	}
}

map<int, Cluster*> kmeans(vector<vector<int>> &points, int kcluster, int iterNum) {
	vector<Point*> newPoints = transform(points);
	map<int, Cluster*> index2kclusters = init(newPoints, kcluster);
	map<int, vector<Point*>> cluster2point;
	int i = 0;
	while (i < iterNum) {
		for (auto point : newPoints) {
			int index = -1;
			double dis = INT_MAX;
			for (auto kv : index2kclusters) {
				Cluster *cluster = kv.second;
				int cIndex = kv.first;
				double distance = computeDis(point, cluster);
				if (distance < dis) {
					dis = distance;
					index = cIndex;
				}
			}
			cluster2point[index].push_back(point);
		}
		updateCluster(cluster2point, index2kclusters);
		i++;
	}
	return index2kclusters;
}

int main()
{
	vector<vector<int>> points = { {0,0},{2,0},{2,2},{0,2},{1,1},{10,5},{12,5},{12,7},{10,7},{11,6},{5,20},{5,22},{7,20},{7,22},{6,21} };
	map<int, Cluster*> clusters = kmeans(points, 3, 6);
	for (auto kv : clusters) {
		cout << kv.first << ": " << kv.second->x << " " << kv.second->y << endl;
	}
    return 0;
}

