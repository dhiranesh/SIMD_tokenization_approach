#include "arena.h"

thread_local Arena ThreadLocalArena::instance(65536);
