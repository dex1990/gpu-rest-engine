#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
typedef struct classifier_ctx classifier_ctx;

typedef struct classifier_ctxlist classifier_ctxlist;

classifier_ctxlist* classifier_initialize(char* model_file, char* trained_file,
                                      char* mean_file, char* label_file);

const char* classifier_classify(classifier_ctxlist* ctxlist,char* buffer);

void classifier_destroy(classifier_ctxlist* ctxlist);

#ifdef __cplusplus
}
#endif

#endif // CLASSIFICATION_H
