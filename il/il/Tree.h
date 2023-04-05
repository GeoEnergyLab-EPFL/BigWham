//==============================================================================
//
// Copyright 2018 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#ifndef IL_TREE_H
#define IL_TREE_H

#include <iostream>
// <cstring> is needed for memcpy
#include <cstring>
// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>
// <new> is needed for placement new
#include <new>
// <utility> is needed for std::move
#include <utility>

#include <il/Array.h>
#include <il/SmallArray.h>
#include "StaticArray.h"


namespace il {

template <typename T, il::int_t n>
class Tree {
 private:
  struct Node {
   public:
    bool empty;
    T value;
    il::int_t parent;
    il::StaticArray<il::int_t, n> children;

   public:
    Node() : empty{}, value{}, parent{-1}, children{-1} {};
  };
  il::int_t depth_;
  il::Array<Node> tree_;

  il::int_t depth_rec(il::spot_t s) ;
  il::int_t nnodesAtLevel_rec(il::spot_t s,il::int_t currentlevel,il::int_t level) const;

  void getNodesAtLevel_rec(il::spot_t s,il::int_t currentlevel,il::int_t level,
                           il::io_t,il::Array<T> &listAtLevel, il::int_t& off_set_k) const;

 public:
  Tree();
  il::int_t depth() const ;
  void setDepth()  ;
  il::spot_t root() const;
  il::spot_t parent(il::spot_t s) const;
  il::spot_t child(il::spot_t s, il::int_t i) const;
  bool hasChild(il::spot_t s) const;
  bool hasChild(il::spot_t s, il::int_t i) const;
  void Set(il::spot_t s, T value);
  void AddChild(il::spot_t s, il::int_t i);
  const T& value(il::spot_t) const;

  il::int_t nnodesAtLevel(il::int_t level) const;
  il::Array<T> getNodesAtLevel(il::int_t level) const;
};

template <typename T, il::int_t n>
Tree<T, n>::Tree() : depth_{0}, tree_{1} {}

template <typename T, il::int_t n>
il::int_t Tree<T, n>::depth() const  {
  return depth_;
}

template <typename T, il::int_t n>
void Tree<T, n>::setDepth()   {
  if (depth_ == 0)
  {
    depth_=this->depth_rec(il::spot_t{0});
  }
}

template <typename T, il::int_t n>
il::int_t Tree<T, n>::depth_rec(il::spot_t s)  {
  if (hasChild(s) == false) {
    return 0;
  } else {
    il::int_t max = 0;
    for (il::int_t i = 0; i < n; i++) {
      const il::spot_t st = this->child(s, i);
      il::int_t aux = this->depth_rec(st);
      if (aux > max) {
        max = aux;
      }
    }
    return (max+1);
  }
}

template <typename T, il::int_t n>
il::spot_t Tree<T, n>::root() const {
  return il::spot_t{0};
};

template <typename T, il::int_t n>
il::spot_t Tree<T, n>::parent(il::spot_t s) const {
  IL_EXPECT_MEDIUM(tree_[s.index].parent >= 0);

  return il::spot_t{tree_[s.index].parent};
};

template <typename T, il::int_t n>
il::spot_t Tree<T, n>::child(il::spot_t s, il::int_t i) const {
  IL_EXPECT_MEDIUM(tree_[s.index].children[i] >= 0);

  return il::spot_t{tree_[s.index].children[i]};
};

template <typename T, il::int_t n>
void Tree<T, n>::Set(il::spot_t s, T x) {
  tree_[s.index].empty = false;
  tree_[s.index].value = x;
}

template <typename T, il::int_t n>
bool Tree<T, n>::hasChild(il::spot_t s) const {
  bool ans = false;
  for (il::int_t i = 0; i < n; ++i) {
    if (tree_[s.index].children[i] >= 0) {
      ans = true;
    }
  }
  return ans;
};

template <typename T, il::int_t n>
bool Tree<T, n>::hasChild(il::spot_t s, il::int_t i) const {
  return tree_[s.index].children[i] >= 0;
};

template <typename T, il::int_t n>
void Tree<T, n>::AddChild(il::spot_t s, il::int_t i) {
  const il::int_t k = tree_.size();
  tree_.Append({});
  tree_[s.index].children[i] = k;
};

template <typename T, il::int_t n>
const T& Tree<T, n>::value(il::spot_t s) const {
  return tree_[s.index].value;
};


template <typename T, il::int_t n>
  il::int_t Tree<T, n>::nnodesAtLevel(il::int_t level) const {
  return nnodesAtLevel_rec(this->root(),0,level);
}


template <typename T, il::int_t n>
il::int_t Tree<T, n>::nnodesAtLevel_rec(il::spot_t s,il::int_t currentlevel,il::int_t level) const {
    if (currentlevel==level) {
      return n;
    }
    il::int_t naux=0;
    for (il::int_t i=0;i<n;i++){
      const il::spot_t st = this->child(s, i);
      naux+=nnodesAtLevel_rec(st,currentlevel+1,level);
    }
    return naux;
}


template <typename T, il::int_t n>
il::Array<T> Tree<T, n>::getNodesAtLevel(il::int_t level) const {

  il::int_t nn=nnodesAtLevel(level);
  il::Array<T> nodesEntryAtL{nn};
  il::int_t offset_i=0;

  getNodesAtLevel_rec(this-> root(),0,level,il::io,nodesEntryAtL,offset_i);
  std::cout << " rec done nodes level!\n";
  return nodesEntryAtL;
}

template <typename T, il::int_t n>
void Tree<T, n>::getNodesAtLevel_rec(il::spot_t s,il::int_t currentlevel,il::int_t level,il::io_t,il::Array<T> &listAtLevel, il::int_t& off_set_k) const {
    if (currentlevel==level-1){
      for (il::int_t i=0;i<n;i++){
        const il::spot_t st = this->child(s, i);
        listAtLevel[off_set_k]=this->value(st);
        off_set_k++;
      }
    } else if (currentlevel<level-1)
    {
      for (il::int_t i=0;i<n;i++){
        const il::spot_t st = this->child(s, i);
        getNodesAtLevel_rec(st,currentlevel+1,level,il::io,listAtLevel,off_set_k);
      }
    } else{

    };

};





}


#endif  // IL_TREE_H
