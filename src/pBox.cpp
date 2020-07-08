#include"pBox.h"

void freepBox(struct pBox *pbox) {
    if (pbox->pdata == NULL)cout << "pbox is NULL!" << endl;
    else
        free(pbox->pdata);
    pbox->pdata = NULL;
    delete pbox;
}

void freeWeight(struct Weight *weight) {
    if (weight->pdata == NULL)cout << "weight is NULL!" << endl;
    else
        free(weight->pdata);
    weight->pdata = NULL;
    delete weight;
}

void freeBN(struct BN *bn) {
    if (bn->pdata == NULL)cout << "weight is NULL!" << endl;
    else
        free(bn->pdata);
    bn->pdata = NULL;
    delete bn;
}
