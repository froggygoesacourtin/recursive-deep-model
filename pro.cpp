#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>

using namespace std;

#define StringLength 200
#define BufLength 2000
#define ElementLength 500000
#define ReviewCount 10000
#define WordCount 20000
#define TrainLength 2000
#define VectorLength 100
#define VectorLength2  (2 * VectorLength)
#define ClassLength 5

float Ws[ClassLength][VectorLength];

float V[VectorLength][VectorLength2][VectorLength2];
float W[VectorLength][VectorLength2];

float tmpS[VectorLength2][VectorLength2];
float S[VectorLength2];
float tmpVector2[VectorLength2];

unordered_map <string, int> wordMap;
int id = 0;

int reviews[ReviewCount];
int reviewCount = 0;

float wordVectors[WordCount][VectorLength];

float tmpVectors[TrainLength][VectorLength];
int tmpCount = 0;

struct Element {
 int _left;
 int _right;
 int _value;
};

struct TrainElement {
 int _left;
 int _right;
 float *_vector;
 float _classVector[ClassLength];
 int _actualValue;
};

char buf[BufLength];
int bufPtr = 0;

struct Element elements[ElementLength];
int elementLen = 0;

struct TrainElement trainElements[TrainLength];
int trainElementLen = 0;

int parse() {
 string str;
 char s[StringLength];
 int ret, left, right, value, i;
 char c;

 if (buf[bufPtr++] == '(') {
  value = (buf[bufPtr++] - '0');

  // eat the space
  bufPtr++;

  if (buf[bufPtr] == '(') {
   left = parse();

   // eat the space
   bufPtr++;

   right = parse();

   elements[elementLen]._left = left;
   elements[elementLen]._right = right;
   elements[elementLen]._value = value;
  }
  else {
   i = 0;
   while (buf[bufPtr] != ' ' && buf[bufPtr] != ')') {
    c = buf[bufPtr];

    // convert to lower case
    if (c >= 'a' && c <= 'z') {
     c -= 'a' - 'A';
    }

    s[i] = c;
    bufPtr++;
    i++;
   }

   s[i] = '\0';
   str = string(s);

   unordered_map<std::string, int>::const_iterator it = wordMap.find(str);
   if ( it == wordMap.end()) {
    wordMap[str] = id;
    left = id;
    id++;
   }
   else {
    left = it->second;
   }

   elements[elementLen]._left = left;
   elements[elementLen]._right = -1;
   elements[elementLen]._value = value;
  }

  // eat the right paren
  bufPtr++;
 }

 ret = elementLen;
 elementLen++;

 return ret;
}

void init() {
 int i, j, k;

 for (i = 0; i < id; i++) {
  for (j = 0; j < VectorLength; j++) {
   wordVectors[i][j] = (.002 * drand48()) - .001;
  }
 }

 for (i = 0; i < ClassLength; i++) {
  for (j = 0; j < VectorLength; j++) {
   Ws[i][j] = (.0002 * drand48()) - .0001;
  }
 }

 for (i = 0; i < VectorLength; i++) {
  for (j = 0; j < VectorLength2; j++) {
   for (k = 0; k < VectorLength2; k++) {
    V[i][j][k] = (.0002 * drand48()) - .0001;
   }
  }
 }

 for (i = 0; i < VectorLength; i++) {
  for (j = 0; j < VectorLength2; j++) {
   W[i][j] = (.0002 * drand48()) - .0001;
  }
 }
}

float softMax(float a[], float s[]) {
 int i, j;
 float d, ret;

 for (i = 0; i < ClassLength; i++) {
  s[i] = 0.;
  for (j = 0; j < VectorLength; j++) {
   s[i] += Ws[i][j] * a[j];
  }
 }

 d = 0;
 for (i = 0; i < ClassLength; i++) {
  d += exp(s[i]);
 }

 ret = 0.;
 for (i = 0; i < ClassLength; i++) {
  ret += i * (exp(s[i]) / d);
 }

 return ret;
}

void comp(float a[], float b[], float c[]) {
 float base[VectorLength2];
 int i, j, k;
 float f, t;

 // Create the compound vector
 for (i = 0; i < VectorLength; i++) {
  base[i] = a[i];
  base[VectorLength + i] = b[i];
 }

 // Derive the tensor product 
 for (i = 0; i < VectorLength; i++) {
  t = 0.;
  for (j = 0; j < VectorLength2; j++) {
   f = 0.;
   for (k = 0; k < VectorLength2; k++) {
    f += V[i][j][k] * base[k];
   }

   t += base[j] * f;
  }

  c[i] = t;
 }

 // Derive the standard layer
 for (i = 0; i < VectorLength; i++) {
  t = 0.;
  for (j = 0; j < VectorLength2; j++) {
   t += W[i][j] * base[j];
  }

  c[i] += t;
 }

 // Derive the non-linearity
 for (i = 0; i < VectorLength; i++) {
  c[i] = tanh(c[i]);
 }
}

int predictTree(int root) {
 float *vector;
 int left, right, ret;

 if (elements[root]._right == -1) {
  vector = wordVectors[elements[root]._left];
  trainElements[trainElementLen]._left = trainElements[trainElementLen]._right = -1;
  trainElements[trainElementLen]._vector = vector;
  softMax(vector, trainElements[trainElementLen]._classVector);
  trainElements[trainElementLen]._actualValue = elements[root]._value;
 }
 else {
  left = predictTree(elements[root]._left);
  trainElements[trainElementLen]._left = left;

  right = predictTree(elements[root]._right);
  trainElements[trainElementLen]._right = right;

  trainElements[trainElementLen]._vector = tmpVectors[tmpCount];

  comp(trainElements[left]._vector, trainElements[right]._vector, tmpVectors[tmpCount]);
  softMax(tmpVectors[tmpCount], trainElements[trainElementLen]._classVector);
  trainElements[trainElementLen]._actualValue = elements[root]._value;

  tmpCount++;
 }

 ret = trainElementLen;
 trainElementLen++;

 return ret;
}

void learnTree(int root, float errParentVector[VectorLength], float errParentV[VectorLength]) {
 float actualClassVector[ClassLength];
 float softMaxErr[VectorLength];
 float errV[VectorLength];
 float childErr2[VectorLength2];
 float *predClassVector;
 float *vector;
 float *leftVector;
 float *rightVector;
 float derivative, td, v;
 int left, right, i, j, k;

 // Compute the softmax error vector
 for (i = 0; i < ClassLength; i++) {
  actualClassVector[i] = 0.;
 }

 actualClassVector[trainElements[root]._actualValue] = 1.;

 predClassVector = trainElements[root]._classVector;
 vector = trainElements[root]._vector;

 for (i = 0; i < VectorLength; i++) {
  softMaxErr[i] = 0.;
  for (j = 0; j < ClassLength; j++) {
   softMaxErr[i] +=  Ws[j][i] * (actualClassVector[j] - predClassVector[j]);
  }

  derivative = tan(vector[i]);
  derivative = 1 - (derivative * derivative);
  softMaxErr[i] *= derivative;
 }

 if (errParentVector != NULL) {
  softMaxErr[i] += errParentVector[i];
 }

 // Apply learning to Ws
 for (i = 0; i < ClassLength; i++) {
  for (j = 0; j < VectorLength; j++) {
   Ws[i][j] +=  Ws[i][j] * (actualClassVector[i] - predClassVector[i]);
  }
 }

 // Is this a non-leaf node
 if (trainElements[root]._left != -1) {
  left = trainElements[root]._left;
  right = trainElements[root]._right;

  leftVector = trainElements[left]._vector;
  rightVector = trainElements[right]._vector;

  // Derive vector component of tensor derivative
  td = 0;
  for (i = 0; i < VectorLength; i++) {
   v = leftVector[i];
   td += v * v;

   v = rightVector[i];
   td += v * v;

   errV[i] = softMaxErr[i] * td;

   if (errParentV != NULL) {
    errV[i] += errParentV[i];
   }
  }

  // derive S
  for (i = 0; i < VectorLength2; i++) {
   tmpVector2[i] = leftVector[i];
   tmpVector2[VectorLength + i] = rightVector[i];
  }

  for (i = 0; i < VectorLength2; i++) {
   for (j = 0; j < VectorLength2; j++) {
    tmpS[i][j] = 0.;
   }
  }

  for (k = 0; k < VectorLength; k++) {
   for (i = 0; i < VectorLength2; i++) {
    for (j = 0; j < VectorLength2; j++) {
     tmpS[i][j] += softMaxErr[k] * (V[k][i][j] + V[k][j][i]);
    }
   }
  }

  for (i = 0; i < VectorLength2; i++) {
   S[i] = 0.;
   for (j = 0; j < VectorLength2; j++) {
    S[i] += tmpS[i][j] * tmpVector2[j];
   }
  }

  // Derive child error
  for (i = 0; i < VectorLength2; i++) {
   childErr2[i] = 0.;
   for (j = 0; j < VectorLength; j++) {
    childErr2[i] += (W[j][i] * softMaxErr[i]);
   }
  }
  
  for (i = 0; i < VectorLength2; i++) {
   derivative = tan(tmpVector2[i]);
   derivative = 1 - (derivative * derivative);

   childErr2[i] += S[i] * derivative;
  }

  // Apply learning to V
  for (i = 0; i < VectorLength; i++) {
   derivative = errV[i];
   for (j = 0; j < VectorLength2; j++) {
    for (k = 0; k < VectorLength2; k++) {
     V[i][j][k] += derivative * V[i][j][k];
    }
   }
  }

  // Apply learning to W
  for (i = 0; i < VectorLength; i++) {
   for (j = 0; j < VectorLength2; j++) {
    W[i][j] += softMaxErr[i] * vector[i];
   }
  }

  learnTree(left, childErr2, errV);
  learnTree(right, &childErr2[VectorLength], errV);
 }
 else {
  // If this is a leaf, apply learning to word vector
  for (i = 0; i < VectorLength; i++) {
   vector[i] += vector[i] * softMaxErr[i];
  }
 }
}

void train(int iterCount) {
 float *classVector;
 int tmpRoot, i, j, k;
 double RMSE, sumErr, pred, err;

 for (i = 0; i < iterCount; i++) {
  sumErr = 0.;
  for (j = 0; j < reviewCount; j++) {
   tmpCount = 0;
   trainElementLen = 0;
   tmpRoot = predictTree(reviews[j]);

   // Compute the error
   classVector = trainElements[tmpRoot]._classVector;
   pred = 0;
   for (k = 0; k < ClassLength; k++) {
    pred += k * classVector[k];
   }

   err = elements[reviews[j]]._value - pred;
   sumErr += err * err;

   learnTree(tmpRoot, NULL, NULL);

   if (j > 0 && j % 1000 == 0) {
    RMSE = sqrt(sumErr / j);
    printf("review id: %d RMSE: %lf\n", j, RMSE);
   }
  }

  RMSE = sqrt(sumErr / reviewCount);
  printf("iteration: %d RMSE: %lf\n", i, RMSE);
 }
}

int main (void) {
 FILE *fp = fopen("train.txt", "r");
 int i;

 srand48(42);

 bufPtr = 0;
 while (fgets (buf, sizeof(buf), fp)) {
  reviews[reviewCount] = parse();  
  reviewCount++;

  bufPtr = 0;
 }

 fclose(fp);

 printf("id: %d\n", id);

 train(50);

 return 0;
}
