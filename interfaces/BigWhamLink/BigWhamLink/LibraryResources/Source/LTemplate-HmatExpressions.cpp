/*  This file was automatically generated by LTemplate. DO NOT EDIT.  */
/*  https://github.com/szhorvat/LTemplate  */

#define LTEMPLATE_MMA_VERSION  1300

#include "LTemplate.h"
#include "LTemplateHelpers.h"
#include "Manager.h"
#include "HMatExpr.h"


#define LTEMPLATE_MESSAGE_SYMBOL  "BigWhamLink`Hmat"

#include "LTemplate.inc"


std::map<mint, Manager *> Manager_collection;

namespace mma
{
	template<> const std::map<mint, Manager *> & getCollection<Manager>()
	{
		return Manager_collection;
	}
}

DLLEXPORT void Manager_manager_fun(WolframLibraryData libData, mbool mode, mint id)
{
	if (mode == 0) { // create
	  Manager_collection[id] = new Manager();
	} else {  // destroy
	  if (Manager_collection.find(id) == Manager_collection.end()) {
	    libData->Message("noinst");
	    return;
	  }
	  delete Manager_collection[id];
	  Manager_collection.erase(id);
	}
}

extern "C" DLLEXPORT int Manager_get_collection(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::TensorRef<mint> res = mma::detail::get_collection(Manager_collection);
	mma::detail::setTensor<mint>(Res, res);
	return LIBRARY_NO_ERROR;
}


std::map<mint, HMatExpr *> HMatExpr_collection;

namespace mma
{
	template<> const std::map<mint, HMatExpr *> & getCollection<HMatExpr>()
	{
		return HMatExpr_collection;
	}
}

DLLEXPORT void HMatExpr_manager_fun(WolframLibraryData libData, mbool mode, mint id)
{
	if (mode == 0) { // create
	  HMatExpr_collection[id] = new HMatExpr();
	} else {  // destroy
	  if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) {
	    libData->Message("noinst");
	    return;
	  }
	  delete HMatExpr_collection[id];
	  HMatExpr_collection.erase(id);
	}
}

extern "C" DLLEXPORT int HMatExpr_get_collection(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::TensorRef<mint> res = mma::detail::get_collection(HMatExpr_collection);
	mma::detail::setTensor<mint>(Res, res);
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT mint WolframLibrary_getVersion()
{
	return WolframLibraryVersion;
}

extern "C" DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData)
{
	mma::libData = libData;
	{
		int err;
		err = (*libData->registerLibraryExpressionManager)("Manager", Manager_manager_fun);
		if (err != LIBRARY_NO_ERROR) return err;
	}
	{
		int err;
		err = (*libData->registerLibraryExpressionManager)("HMatExpr", HMatExpr_manager_fun);
		if (err != LIBRARY_NO_ERROR) return err;
	}
	return LIBRARY_NO_ERROR;
}

extern "C" DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData)
{
	(*libData->unregisterLibraryExpressionManager)("Manager");
	(*libData->unregisterLibraryExpressionManager)("HMatExpr");
	return;
}


extern "C" DLLEXPORT int Manager_releaseHMatExpr(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (Manager_collection.find(id) == Manager_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mint var1 = MArgument_getInteger(Args[1]);
		
		(Manager_collection[id])->releaseHMatExpr(var1);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "Manager::releaseHMatExpr()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "Manager::releaseHMatExpr()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_set(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mma::TensorRef<double> var1 = mma::detail::getTensor<double>(Args[1]);
		mma::TensorRef<mint> var2 = mma::detail::getTensor<mint>(Args[2]);
		const char * var3 = mma::detail::getString(Args[3]);
		mma::TensorRef<double> var4 = mma::detail::getTensor<double>(Args[4]);
		mint var5 = MArgument_getInteger(Args[5]);
		double var6 = MArgument_getReal(Args[6]);
		double var7 = MArgument_getReal(Args[7]);
		
		(HMatExpr_collection[id])->set(var1, var2, var3, var4, var5, var6, var7);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::set()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::set()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_isBuilt(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		bool res = (HMatExpr_collection[id])->isBuilt();
		MArgument_setBoolean(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::isBuilt()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::isBuilt()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_getKernel(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		const char * res = (HMatExpr_collection[id])->getKernel();
		mma::detail::setString(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::getKernel()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::getKernel()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_getPermutation(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mma::TensorRef<mint> res = (HMatExpr_collection[id])->getPermutation();
		mma::detail::setTensor<mint>(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::getPermutation()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::getPermutation()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_getCollocationPoints(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mma::TensorRef<double> res = (HMatExpr_collection[id])->getCollocationPoints();
		mma::detail::setTensor<double>(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::getCollocationPoints()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::getCollocationPoints()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_getCompressionRatio(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		double res = (HMatExpr_collection[id])->getCompressionRatio();
		MArgument_setReal(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::getCompressionRatio()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::getCompressionRatio()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_getHpattern(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mma::TensorRef<mint> res = (HMatExpr_collection[id])->getHpattern();
		mma::detail::setTensor<mint>(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::getHpattern()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::getHpattern()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_hdot(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mma::TensorRef<double> var1 = mma::detail::getTensor<double>(Args[1]);
		
		mma::TensorRef<double> res = (HMatExpr_collection[id])->hdot(var1);
		mma::detail::setTensor<double>(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::hdot()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::hdot()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_getSize(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mma::TensorRef<mint> res = (HMatExpr_collection[id])->getSize();
		mma::detail::setTensor<mint>(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::getSize()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::getSize()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_getProblemDimension(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mint res = (HMatExpr_collection[id])->getProblemDimension();
		MArgument_setInteger(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::getProblemDimension()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::getProblemDimension()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_getFullBlocks(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mma::SparseArrayRef<double> res = (HMatExpr_collection[id])->getFullBlocks();
		mma::detail::setSparseArray<double>(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::getFullBlocks()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::getFullBlocks()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


extern "C" DLLEXPORT int HMatExpr_computeStresses(WolframLibraryData libData, mint Argc, MArgument * Args, MArgument Res)
{
	mma::detail::MOutFlushGuard flushguard;
	const mint id = MArgument_getInteger(Args[0]);
	if (HMatExpr_collection.find(id) == HMatExpr_collection.end()) { libData->Message("noinst"); return LIBRARY_FUNCTION_ERROR; }
	
	try
	{
		mma::TensorRef<double> var1 = mma::detail::getTensor<double>(Args[1]);
		mma::TensorRef<double> var2 = mma::detail::getTensor<double>(Args[2]);
		mint var3 = MArgument_getInteger(Args[3]);
		mma::TensorRef<double> var4 = mma::detail::getTensor<double>(Args[4]);
		mma::TensorRef<double> var5 = mma::detail::getTensor<double>(Args[5]);
		mma::TensorRef<mint> var6 = mma::detail::getTensor<mint>(Args[6]);
		bool var7 = MArgument_getBoolean(Args[7]);
		
		mma::TensorRef<double> res = (HMatExpr_collection[id])->computeStresses(var1, var2, var3, var4, var5, var6, var7);
		mma::detail::setTensor<double>(Res, res);
	}
	catch (const mma::LibraryError & libErr)
	{
		libErr.report();
		return libErr.error_code();
	}
	catch (const std::exception & exc)
	{
		mma::detail::handleUnknownException(exc.what(), "HMatExpr::computeStresses()");
		return LIBRARY_FUNCTION_ERROR;
	}
	catch (...)
	{
		mma::detail::handleUnknownException(NULL, "HMatExpr::computeStresses()");
		return LIBRARY_FUNCTION_ERROR;
	}
	
	return LIBRARY_NO_ERROR;
}


