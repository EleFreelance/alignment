#include"NNF.h"

NNF_P initNNF(Mat _ref, Mat _match, int _patch_radius)
{
	//输入参数检验！
	NNF_P p_nnf = new NNF();
	p_nnf->ref = _ref;
	p_nnf->match = _match;
	p_nnf->patch_radius = _patch_radius;

	p_nnf->field = NULL;
	p_nnf->fieldH = _ref.rows;
	p_nnf->fieldW = _ref.cols;
	return p_nnf;
}

void mallocFiled(NNF_P p_nnf)
{
	if (p_nnf->field == NULL)
	{
		p_nnf->field = (int ***)malloc(sizeof(int**)*p_nnf->fieldH);
		for (int y = 0; y < p_nnf->fieldH; y++)
		{
			p_nnf->field[y] = (int**)malloc(sizeof(int*)*p_nnf->fieldW);
			for (int x = 0; x < p_nnf->fieldW; x++)
			{
				p_nnf->field[y][x] = (int*)malloc(sizeof(int) * 3);
			}
		}
	}
}

void freeFiled(NNF_P p_nnf)
{
	if (p_nnf -> field != NULL)
	{
		for (int y = 0; y < p_nnf->fieldH; y++)
		{
			for (int x = 0; x < p_nnf->fieldW; x++)
			{
				free(p_nnf->field[y][x]);
			}
			free(p_nnf->field[y]);
		}
		free(p_nnf->field);
	}
}

void randomize(NNF_P p_nnf)
{
	mallocFiled(p_nnf);
	for (int y = 0; y < p_nnf->fieldH; y++)
	{
		for (int x = 0; x < p_nnf->fieldW; x++)
		{
			p_nnf->field[y][x][0] = rand() % p_nnf->fieldW;
			p_nnf->field[y][x][1] = rand() % p_nnf->fieldH;
			p_nnf->field[y][x][2] = MAX_DISTANCE;
		}
	}
	initialization(p_nnf);
}

void initialization(NNF_P p_nnf)
{
	int max_iter = 20;
	int iter = 0;
	for (int y = 0; y < p_nnf->fieldH; y++)
	{
		for (int x = 0; x < p_nnf->fieldW; x++)
		{
			p_nnf->field[y][x][2] = distance(p_nnf, x, y, p_nnf->field[y][x][0], p_nnf->field[y][x][1]);
			iter = 0;
			while (iter < max_iter&&p_nnf->field[y][x][2] == MAX_DISTANCE)
			{
				p_nnf->field[y][x][0] = rand() % p_nnf->fieldW;
				p_nnf->field[y][x][1] = rand() % p_nnf->fieldH;
				p_nnf->field[y][x][2] = distance(p_nnf, x, y,  p_nnf->field[y][x][0], p_nnf->field[y][x][1]);
				iter++;
			}
		}
	}
}

void minimize(NNF_P p_nnf, int max_iter)
{
	int start_x, end_x, start_y, end_y, dir;
	if (max_iter % 2 == 0)
	{
		start_x = 0;
		end_x = p_nnf->fieldW;
		start_y = 0;
		end_y = p_nnf->fieldH;
		dir = 1;
	}
	else
	{
		start_x = p_nnf->fieldW - 1;
		end_x = -1;
		start_y = p_nnf->fieldH - 1;
		end_y = -1;
		dir = -1;
	}

	for (int y = start_y; y*dir < end_y*dir; y+=dir)
	{
		for (int x = start_x; x*dir < end_x*dir; x += dir)
		{
			if (p_nnf->field[y][x][2] > 0)
				minimizeLink(p_nnf, x, y, dir);
		}
	}
}

void minimizeLink(NNF_P p_nnf,int x, int y, int dir)
{
	//propagation
	//top,bottom
	if (y - dir > 0 && y - dir < p_nnf->fieldH)
	{
		int xp = p_nnf->field[y - dir][x][0];
		int yp = p_nnf->field[y - dir][x][1] + dir;
		int dp = distance(p_nnf, x, y, xp, yp);
		if (dp < p_nnf->field[y][x][2])
		{
			p_nnf->field[y][x][0] = xp;
			p_nnf->field[y][x][1] = yp;
			p_nnf->field[y][x][2] = dp;
		}
	}

	//left,right
	if (x- dir > 0 && x - dir < p_nnf->fieldW)
	{
		int xp = p_nnf->field[y][x - dir][0] + dir;
		int yp = p_nnf->field[y][x - dir][1];
		int dp = distance(p_nnf, x, y, xp, yp);
		if (dp < p_nnf->field[y][x][2])
		{
			p_nnf->field[y][x][0] = xp;
			p_nnf->field[y][x][1] = yp;
			p_nnf->field[y][x][2] = dp;
		}
	}
	//random search
	int wi = p_nnf->fieldW;
	int xpi = p_nnf->field[y][x][0];
	int ypi = p_nnf->field[y][x][1];
	while (wi > 0)
	{
		int xp = xpi + rand() % 2 * wi - wi;
		int yp = ypi + rand() % 2 * wi - wi;

		xp = min(max(0, xp), p_nnf->fieldW);
		yp = min(max(0, yp), p_nnf->fieldH);
		int dp = distance(p_nnf, x, y, xp, yp);
		if (dp < p_nnf->field[y][x][2])
		{
			p_nnf->field[y][x][0] = xp;
			p_nnf->field[y][x][1] = yp;
			p_nnf->field[y][x][2] = dp;
		}
		wi /= 2;
	}
}


int distance(NNF_P p_nnf, int xs, int ys, int xt, int yt)
{
	uchar *ptr_ref = p_nnf->ref.data;
	uchar *ptr_match = p_nnf->match.data;

	int s = ptr_ref[ys*p_nnf->fieldW + xs];
	int t = ptr_match[yt*p_nnf->fieldW + xt];

	return (s - t)*(s - t);
}