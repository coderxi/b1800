#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <vector>
#include <deque>
#include <iterator>
#include <math.h>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

// Mat(rows, cols), cols major order. (col by col append), new format
cv::Mat_<double> id_load(char * file)
{
    std::ifstream in(file);
    int i, j, cols, rows;
    double dummy;
    in >> cols >> rows >> dummy;
    Mat_<double> mat(rows,cols);
    double d;
	for (j=0; j<rows; ++j)
    for (i=0; i<cols; ++i) {
        in >> d;
        mat.at<double>(Point(i,j)) = d;
	}
    // return mat.t();
    return mat;
}

template <typename F>
void each(int n, F && f)
{
    for (int i=0; i<n; ++i)
        f(i);
}

template <typename F>
void each(int h, int w, F && f)
{
    for (int j=0; j<w; ++j)
    for (int i=0; i<h; ++i)
        f(i,j);
}

template <typename T, typename F>
void each(Mat_<T> & mat, int x0, int x1, int y0, int y1, F && f)
{
    x0 = x0<0 ? 0 : x0;
    y0 = y0<0 ? 0 : y0;
    x1 = x1>mat.rows ? mat.rows : x1;
    y1 = y1>mat.cols ? mat.cols : y1;
    for (int y=y0; y<y1; ++y)
    for (int x=x0; x<x1; ++x)
        f(x,y);
}


struct c_line { 
	Vec2f p[2]; 
	float len() { return (float)cv::norm(p[1] - p[0]); }
	Vec2f mid() { return (p[0] + p[1]) * 0.5; }
	Vec2f dir() { return (p[1] - p[0]) * (1 / len()); }
};

c_line make_line(Vec2f a, Vec2f b) {
    c_line line;
    line.p[0] = a; line.p[1] = b;
    return line;
}

Vec2f pp_dir(Vec2f dir) { return Vec2f(dir[1], -dir[0]); }

c_line to_line(const Vec4i &l) {
	c_line line;
	Vec2i a(&l[0]), b(&l[2]);
	line.p[0] = Vec2f(a);
	line.p[1] = Vec2f(b);
	return line;
}

struct c_seg {
	float x, y;
	c_seg(float a, float b) { x = min(a,b); y = max(a,b); }
} ;

bool merge(c_seg a, c_seg b, float gap, c_seg &seg) {
	if (a.x > b.x)  std::swap(a,b); // make sure a.x <= b.x
	if (b.x - a.y < gap) 
		seg = c_seg(a.x, max(a.y, b.y));    // overlap
	else
		return false;
	return true;
}

class c_lineprocess {
public:
    c_lineprocess(vector<Vec4i> &lines) {
        _lines.resize(lines.size());
        for (unsigned int i=0; i<lines.size(); ++i)
            _lines[i] = to_line(lines[i]);
	}

    // all lines must align with primary directions
    int p10_rectify(std::vector<Vec2f> & pdirs, float thd) {
        std::vector<c_line> lines;
        lines.reserve(_lines.size());
        _pdirs = pdirs;
        _dirs.clear();
        _dirs.reserve(_lines.size());
        for (unsigned int i=0; i<_lines.size(); ++i) {
            c_line & l = _lines[i];
            c_line rsl;
            for (unsigned int j=0; j<pdirs.size(); ++j)
                if (fabs(pdirs[j].dot(l.dir())) > thd) { // bingo, find the closed primary direction
                    Vec2f m = l.mid();
                    float len = fabs((l.p[1] - l.p[0]).dot(pdirs[j]));
                    rsl.p[0] = m - len * 0.5 * pdirs[j];
                    rsl.p[1] = m + len * 0.5 * pdirs[j];
                    lines.push_back(rsl);
                    _dirs.push_back(j);
                    break;
				}
		}
        std::swap(_lines, lines);
        assert(_lines.size() == _dirs.size());
        std::cout << "p10_rectify, return " << _lines.size() << "\n";
        return _lines.size();
	}

    // merge line with same direction
    int p20_snapping(float thd1, float thd_gap) {
        std::vector<int> merged(_lines.size(), -1);
        for (unsigned int i=0; i<_lines.size()-1; ++i)
            for (unsigned int j=i+1; j<_lines.size(); ++j) {
                c_line & a = _lines[i];
                c_line & b = _lines[j];
                if (_dirs[i] != _dirs[j])
                    continue;
                Vec2f dir = _pdirs[_dirs[i]];
                Vec2f xdir(dir[1], -dir[0]);    // 90 degree vector
                float rhoa = xdir.dot(a.p[0]);
                float rhob = xdir.dot(b.p[0]);
                if (fabs(rhoa - rhob) > thd1)
                    continue;   //  distance too much 
                c_seg sega(a.p[0].dot(dir), a.p[1].dot(dir));
                c_seg segb(b.p[0].dot(dir), b.p[1].dot(dir));
                c_seg seg(0,0);
                if (!merge(sega, segb, thd_gap, seg))
                    continue;
                // merge i j to j (for further merging)
                float lena = a.len(), lenb = b.len();
                float rho = (lena * rhoa + lenb * rhob) / (lena + lenb);
                b.p[0] = xdir * rho + seg.x * dir;
                b.p[1] = xdir * rho + seg.y * dir;
                merged[i] = j;
			}
        std::vector<c_line> lines;
        std::vector<int> dirs;
        lines.reserve(_lines.size());
        dirs.reserve(_lines.size());
        for (unsigned int i=0; i<_lines.size(); ++i)
            if (merged[i] < 0) {
                lines.push_back(_lines[i]);
                dirs.push_back(_dirs[i]);
			}
        std::swap(_lines, lines);
        std::swap(_dirs, dirs);

        std::cout << "p20_snapping, return " << _lines.size() << "\n";
        return _lines.size();
	}

    // connect end points or nearsst line if distance < r
    // cross threshold 0.1, parallel threshold 0.95
    int p30_cross_connect(float r) {
        std::cout << "p30_cross_connect " << r << "\n";
        std::vector<c_line> lines(_lines);  // original copy
        _cross_conns.clear();
        _cross_conns.resize(lines.size());

        std::cout << "first, connect all cross lines\n";
        for (unsigned int i=0; i<_lines.size(); ++i) {
            c_line & line_i = _lines[i];
			Vec2f dir_i = line_i.dir();
			float ext[2] = {0,0};
            // compare with original lines always
            for (unsigned int j=0; j<lines.size(); ++j) {
                if (i==j)
                    continue;
                if (i==6 && j==8)
                    i = 6;
                c_line & line_j = lines[j];
                Vec2f dir_j = line_j.dir();
                if (dir_i.dot(dir_j) > 0.1) // not perpendicular/cross
                    continue;

				float e = cross_conn(line_i.p[0], -dir_i, r, line_j);
                if (e > ext[0])
                    ext[0] = e;
				e = cross_conn(line_i.p[1], dir_i, r, line_j);
                if (e > ext[1])
                    ext[1] = e;
			}
            std::cout << "line " << i << ", from " << line_i.p[0] << ", " << line_i.p[1];
            Vec2i cc(0,0);
            if (ext[0] > 0) {
                line_i.p[0] -= ext[0] * dir_i;
                cc[0] = 1;
			}
            if (ext[1] > 0) {
                line_i.p[1] += ext[1] * dir_i;
                cc[1] = 1;
			}
            std::cout << ", to " << line_i.p[0] << ", " << line_i.p[1] << "\n";
            _cross_conns[i] = (cc);
		}

        return _lines.size();
	}

    float cross_conn(Vec2f p, Vec2f dir, float r, c_line l) {
        Vec2f xdir = pp_dir(dir), ldir = l.dir(), lxdir = pp_dir(ldir);
        if (lxdir.dot(dir) < 0)
            lxdir = -lxdir;
        float ext = (l.p[0] - p).dot(lxdir) / (lxdir.dot(dir));
        // case 1, too far
        if (ext < 0 || ext > r)
            return 0;
        // case 2, in r radius
        if (cv::norm(p-l.p[0]) < r || cv::norm(p-l.p[1]) < r)
            return ext;
        // case 3, close line,
        if ((l.p[0] - p).dot(xdir) * (l.p[1] - p).dot(xdir) < 0)
            return ext;
        return 0;
	}

    int p40_parallel_connect(float thd) {
        std::cout << "p40_parallel_connect " << thd << "\n";
        // auto lines = _lines;
        _parallel_conns.resize(_lines.size(), Vec2i(0,0));
        int c = _lines.size();
        _lines.reserve(c*2);
        _dirs.reserve(c*2);
        for (int i=0; i<c-1; ++i) {
            c_line &line_i = _lines[i];
            Vec2f dir_i = line_i.dir();
            for (int j=i+1; j<c; ++j) {
                c_line &line_j = _lines[j];
                Vec2f dir_j = line_j.dir();

                if (dir_i.dot(dir_j) < 0.95)     // no parallel nothing to do
                    continue;

                for (int a=0; a<2; ++a) for (int b=0; b<2; ++b) {
                    if (_cross_conns[i][a] || _cross_conns[j][b])   // already connect by cross
                        continue;
                    if (_parallel_conns[i][a] || _parallel_conns[j][b])   // already connect by cross
                        continue;
                    if (cv::norm(line_i.p[a] - line_j.p[b]) < thd)  { // ok connection them
						// parallel it, hack! adjust 
						Vec2f xdir = pp_dir(dir_i);
						Vec2f dir = dir_i;
						Vec2f &A = line_i.p[a];
						Vec2f &B = line_j.p[b];
						Vec2f move = (B-A).dot(dir) * dir * 0.5;
						A += move;
						B -= move;

                        _parallel_conns[i][a] = 1;
                        _parallel_conns[j][b] = 1;
                        _lines.push_back(make_line(A, B));

                        assert(_dirs[i] == _dirs[j]);
                        _dirs.push_back((_dirs[i]+1)%2);
					}
				}
			}
		}

        assert(_dirs.size() == _lines.size());
        std::cout << "set lines pts follwoing dir.\n";
        for (int i=0; i<_lines.size(); ++i) {
            c_line &l = _lines[i];
            int d = _dirs[i];   assert(d==0 || d==1);
            Vec2f dir = _pdirs[d];
            if (dir.dot(l.dir()) < 0)
                std::swap(l.p[0], l.p[1]);
            assert(dir.dot(l.dir()) > 0.95);
		}

        return _lines.size();
	}

    std::vector<c_line> _lines;
    std::vector<Vec2f>  _pdirs;
    std::vector<int>    _dirs;
    std::vector<Vec2i>  _cross_conns;
    std::vector<Vec2i>  _parallel_conns;
};

int id_main( int argc, char** argv )
{
    // Mat_<double> mat = id_load("e:\\tmp\\001\\newData3\\floorDepth.txt");
    Mat_<double> mat = id_load("e:\\tmp\\001\\WeightedDepth.txt");
    // Mat_<double> mat = id_load("e:\\tmp\\001\\data2\\floorDepth.txt");
    // Mat_<double> mat = id_load("e:\\tmp\\001\\realDoorHeight.txt");
    // Mat_<double> mat = id_load("C:/work/tmp/003.rsh/newData/floorDepth.txt");
    int w = mat.cols, h = mat.rows;
    std::cout << "mat.cols " << w << ", mat.rows " << h << ", width " << mat.size().width << ", height " << mat.size().height << std::endl;
	imshow("source", mat);

    double threshold = 0, max_value = 0;
	{
		std::vector<double> vec;
		vec.reserve(w*h);
		each(h, w, [&](int i, int j) { vec.push_back(mat(i,j)); });
		std::sort(vec.begin(), vec.end());
		threshold = vec[(int)(vec.size() * 0.97)];
        max_value = *vec.rbegin();

        std::cout << "ordered image value " << vec.size() << " : \n";
        std::copy(vec.begin(), vec.begin()+10, std::ostream_iterator<double>(std::cout, ", "));
        std::cout << " ... ";
        std::copy(vec.end()-10, vec.end(), std::ostream_iterator<double>(std::cout, ", "));
        std::cout << std::endl;
		std::cout << "set threshold : " << threshold << std::endl;
		std::cout << "max value : " << max_value << std::endl;
	}

    std::cout << "build edge map for hough detection\n";
    Mat_<unsigned char> edge(h,w);
	// each(h, w, [&](int i, int j) { edge(i,j) = mat(i,j) > threshold ? (unsigned char)(255 * mat(i,j)/max_value) : 0; });
	each(h, w, [&](int i, int j) { edge(i,j) = mat(i,j) > threshold ? 255 : 0; });
	imshow("edge", edge);

    std::cout << "hough line segment search\n";
	vector<Vec4i> lines;
	// HoughLinesP(edge, lines, 1, CV_PI/180, 5, 15, 5 );
	HoughLinesP(edge, lines, 1, CV_PI/180, 9, 10, 10 );

    std::cout << "extract and draw line segments\n";
    Mat rsl(h, w, CV_8UC3);
    vector<Vec3f> lines_info(lines.size()); // cos/sin/len
    int num_of_dirs = 0;
	for( size_t i = 0; i < lines.size(); i++ )
	{
		cv::Vec4i l = lines[i];
        cv::Vec3f & info = lines_info[i];
        float dx = (float)(l[2] - l[0]), dy = (float)(l[3] - l[1]);
        float d = sqrt(dx*dx + dy * dy);
        info[0] = dx/d;  // cos(theta)
        info[1] = dy/d;  // sin(theta)
        info[2] = d;     // line seg length
        num_of_dirs += (int)ceil(d);

        // std::cout << "line " << i << " : " << lines[i] << std::endl;
        if (d > 50)
		line( rsl, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
	}

	{
        auto l2 = lines;
        auto li2 = lines_info;
        l2.clear();
        li2.clear();
        for (unsigned int i=0; i<lines.size(); ++i) {
            if (lines_info[i][2] > 50) {
                l2.push_back(lines[i]);
                li2.push_back(lines_info[i]);
			}
		}
        std::swap(lines, l2);
        std::swap(lines_info, li2);
	}

	// each(lines.size(), [&](int i) { 
	// 	std::cout << "line seg " << i << " cos/sin " << lines_info[i] << ", pts " << lines[i] << std::endl; 
	// });
	imshow("result", rsl);

    std::cout << "draw direction circle map\n";
    Mat rsl_dir(800, 800, CV_8UC3);
    each(lines_info.size(), [&](int i) {
        cv::Vec3f &info = lines_info[i];
        int x = (int)(400 + info[2] * info[0]);
        int y = (int)(400 + info[2] * info[1]);
        line(rsl_dir, Point(400,400), Point(x,y), Scalar(0,0,255), 1, CV_AA);
	});

    std::vector<float> vdirs;
	vdirs.reserve(2*num_of_dirs);
    each(lines_info.size(), [&](int i) {
        cv::Vec3f &info = lines_info[i];
        int len = (int)ceil(info[2]);
        if (len < 50) return;
        auto & vd = vdirs;
        each(len, [&](int j) {   // points number is same as length
            vd.push_back(info[0]);
            vd.push_back(info[1]);
		});
	});
    Mat dirs(vdirs.size()/2, 2, CV_32F, &vdirs[0]);
    // address is same
    std::cout << "mat's element address : " << &dirs.at<float>(0,0) << ", vector address : " << &vdirs[0] << "\n";
    Mat labels, centers;
    std::vector<Vec2f> pdirs(2); // primary direction, 2 way
    kmeans(dirs, 2, labels, 
		cv::TermCriteria(CV_TERMCRIT_EPS, 100, 0), 
		10, KMEANS_RANDOM_CENTERS, centers);
    std::cout << "kmean result, labels size " << labels.size() << ", centers : \n" << centers << "\n";
    each(2, [&](int i){
        Vec2f & dir = centers.at<Vec2f>(i, 0);
        dir = dir * (1.0/(dir[0] * dir[0] + dir[1] * dir[1]));
        int x = (int)(400 + 200 * dir[0]);
        int y = (int)(400 + 200 * dir[1]);
		line(rsl_dir, Point(400,400), Point(x,y), Scalar(255,0,0), 1, CV_AA);
        pdirs[i] = dir;
	});
    std::cout << "two direction dot product " << centers.at<Vec2f>(0,0).dot(centers.at<Vec2f>(1,0)) << "\n";
	imshow("rsl_dir", rsl_dir);

    std::cout << "rectify all line segment.\n";
    c_lineprocess lp(lines);
    lp.p10_rectify(pdirs, 0.95f);
    lp.p20_snapping(10, 5);
    lp.p30_cross_connect(40);
    lp.p40_parallel_connect(30);
    Mat rf_rsl(h, w, CV_8UC3);
    std::cout << "draw line : \n";
    for (unsigned int i=0; i<lp._lines.size(); ++i) {
        c_line &l = lp._lines[i];
		line(rf_rsl, Point(l.p[0]), Point(l.p[1]), Scalar(255,0,0), 1, CV_AA);
		std::cout <<  i << "  " << lp._lines[i].p[0] << "  " << lp._lines[i].p[1] << "\n";
		{
            std::stringstream ss;
            ss << i;
			string text = ss.str();
			int fontFace = FONT_HERSHEY_SIMPLEX;
			double fontScale = 0.5;
			int thickness = 1;  
			// cv::Point textOrg(5, 65);
			cv::Point textOrg = (Point(l.p[0]) + Point(l.p[1])) * 0.5;
			cv::putText(rf_rsl, text, textOrg, fontFace, fontScale, Scalar(0,0,255), thickness,2);
		}
	}
	imshow("50 - rectify", rf_rsl);

    // hard code door, and build mesh, obj file
	{
        std::cout << "build obj file\n";
        std::map<int, cv::Vec2f> doors;
        doors[0] = Vec2f(0.3f, 1);
        doors[8] = Vec2f(0, 0.5f);
        doors[13] = Vec2f(0.85f, 1);
        float z0 = -2.13, z1 = 3.4, zd = 0.84;
        float xunit = 17.831 / 606, yunit = 15.0699 / 512;
        float x0 = -13.5105, y0 = -9.68499;

        std::vector<c_line> &lines = lp._lines;
        std::deque<Vec3f> pts;
        std::deque<Vec4i> rects;

#define RECT(p, q, z, w) do { \
    int i = pts.size(); \
    pts.push_back(Vec3f(p[0], p[1], z)); \
    pts.push_back(Vec3f(q[0], q[1], z)); \
    pts.push_back(Vec3f(q[0], q[1], w)); \
    pts.push_back(Vec3f(p[0], p[1], w)); \
	rects.push_back(Vec4i(i,i+1,i+2,i+3)); } while (0)

        for (int i=0; i<lines.size(); ++i) {
            c_line &l = lines[i];
            Vec2f a = l.p[0], b = l.p[1], c, d;
            a[0] = x0 + xunit * a[0];
            a[1] = y0 + yunit * a[1];
            b[0] = x0 + xunit * b[0];
            b[1] = y0 + yunit * b[1];
            if (doors.find(i) == doors.end()) // simple case, one rectangle
                RECT(a, b, z0, z1);
			else {  // processing door a -- c -(door)- d -- b
                Vec2f D = doors[i];
                c = a + (b-a) * D[0];
                d = a + (b-a) * D[1];
                RECT(a,c,z0,z1);
                RECT(c,d,z0,zd);
                RECT(d,b,z0,z1);
			}
		}

        std::cout << "total pts : " << pts.size() << ", total rects : " << rects.size() << "\n";

        std::ofstream out("b1800.obj");
        for (int i=0; i<pts.size(); ++i) {
            Vec3f &p = pts[i];
            out << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
		}
        for (int i=0; i<rects.size(); ++i) {
            Vec4i &p = rects[i] + Vec4i(1,1,1,1);
            out << "f " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << "\n";
            // out << "f " << p[0] << " " << p[1] << " " << p[2] << "\n";
            // out << "f " << p[2] << " " << p[3] << " " << p[0] << "\n";
		}
	}

    std::cout << "detect corner\n";
    Mat corner = Mat::zeros( edge.size(), CV_32FC1);
	cornerHarris( edge, corner, 7, 5, 0.05, BORDER_DEFAULT );
    // Normalizing
    Mat corner_norm, corner_norm_scaled;
    normalize( corner, corner_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( corner_norm, corner_norm_scaled );
    std::cout << "get top 1% from corner\n";
    Mat rsl_corner(corner.size(), CV_8UC1);
    std::cout << "corner_norm_scaled, element type : " << corner_norm_scaled.type() << "\n";
	if (1) {
        std::vector<double> vec;
        vec.reserve(w*h);
		each(h, w, [&](int i, int j) { vec.push_back(corner_norm_scaled.at<unsigned char>(i,j)); });
        std::sort(vec.begin(), vec.end());
        double thd = vec[(int)(0.99 * vec.size())];
        double maxv = *vec.rbegin();
		each(h, w, [&](int i, int j) { 
			double c = corner_norm_scaled.at<unsigned char>(i,j);
			rsl_corner.at<unsigned char>(i,j) = c > thd ? (unsigned char)(255 * (c-thd)/(maxv-thd)) : 0;
		});
	}
	imshow("corner_scaled", corner_norm_scaled);
	imshow("rsl_corner", rsl_corner);

    cv::namedWindow( "source", cv::WINDOW_AUTOSIZE ); // Create a window for display.
    cv::waitKey(0); // Wait for a keystroke in the window

    return 0;
} 


// all hard coded
void build_obj()
{
// 0  [430.448, 55.6723]  [519.988, 49.2967]
// 1  [361.919, 94.8421]  [370.095, 177.255]
// 2  [344.73, 31.997]  [351.071, 95.9182]
// 3  [235.494, 127.244]  [259.205, 366.248]
// 4  [48.1153, 452.057]  [105.885, 447.943]
// 5  [544.192, 284.546]  [556.641, 410.028]
// 6  [345.839, 43.1733]  [429.135, 37.2423]
// 7  [108.971, 22.8069]  [115.281, 86.4039]
// 8  [253.301, 366.668]  [264.001, 474.53]
// 9  [30.2519, 491.173]  [578.519, 452.135]
// 10  [66.0204, 380.003]  [259.205, 366.248]
// 11  [344.73, 31.997]  [457.038, 24.0003]
// 12  [517.397, 23.179]  [534.367, 194.234]
// 13  [356.392, 178.615]  [380.591, 422.541]
// 14  [370.181, 419.497]  [580.614, 404.514]
// 15  [27.1052, 91.0098]  [237.775, 76.0094]
// 16  [429.135, 37.2423]  [430.448, 55.6723]
// 17  [351.071, 95.9182]  [361.919, 94.8421]
// 18  [356.392, 178.615]  [370.095, 177.255]
    // hard code door, and build mesh, obj file
	{
        std::map<int, cv::Vec2f> doors;
        doors[0] = Vec2f(0.3f, 1);
        doors[8] = Vec2f(0, 0.5f);
        doors[13] = Vec2f(0.85f, 1);
        float z0 = -2.13, z1 = 3.4, zd = 0.84;
        float xunit = 17.831 / 606, yunit = 15.0699 / 512;
	}
}
