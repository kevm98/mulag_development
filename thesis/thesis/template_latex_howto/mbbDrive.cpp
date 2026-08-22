// this is a -*- C++ -*- file
//----------------------------------------------------------------------
/*!\file    mbbDrive.C
 *
 * \author   Jochen Hirth
 * \date    18.04.06
 *
 * $Revision: 1.3 $
 * $Id: NullBehaviourBasedModule.C,v 1.3 2005/10/04 11:17:39 martin Exp $
 *
 */ 
//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------

#include "mbbDrive.h"

//----------------------------------------------------------------------
// Module Debug
//----------------------------------------------------------------------
//#undef MODULE_DEBUG
#define MODULE_DEBUG
#include "kernel/ModuleDebug.h"

//----------------------------------------------------------------------
// Debugging
//----------------------------------------------------------------------
#undef LOCAL_DEBUG 
//#define LOCAL_DEBUG
#include "kernel/LocalDebug.h"

//----------------------------------------------------------------------
// defines and consts
//----------------------------------------------------------------------

//----------------------------------------------------------------------
// class mbbDrive constructor
//----------------------------------------------------------------------
mbbDrive::mbbDrive( tParent *parent, tDescription name, tBehBBHandler* _beh_bb_handler, bool fixit )
        : tBehaviourBasedModule( parent, name,
                                 eSI_DIMENSION, eSO_DIMENSION, eCI_DIMENSION, eCO_DIMENSION,
                                 si_description, so_description, ci_description, co_description, _beh_bb_handler )
{

    sigmoid = new tSigmoid( tSigmoid::eSIG_RISE, 0., 1. );
}

//----------------------------------------------------------------------
// class mbbDrive destructor
//----------------------------------------------------------------------
mbbDrive::~mbbDrive()
{
    delete sigmoid;
}

