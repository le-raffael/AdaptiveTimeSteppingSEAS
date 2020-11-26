#!/bin/bash

echo "type,dt,V_err,psi_err"
N=10

for i in `seq 0 $N`
do
    dt=`lua -e "print(1.0/(10.0 * 2^$i))"`
    out=`./rs -ts_type euler -ts_adapt_type none -ts_dt ${dt}`
    echo `echo $out | awk '{ print "euler,"'$dt'","$2","$4}'`
done

for i in `seq 0 $N`
do
    dt=`lua -e "print(1.0/(10.0 * 2^$i))"`
    out=`./rs -ts_type rk -ts_rk_type 3bs -ts_adapt_type none -ts_dt ${dt}`
    echo `echo $out | awk '{ print "3bs,"'$dt'","$2","$4}'`
done

for i in `seq 0 $N`
do
    dt=`lua -e "print(1.0/(10.0 * 2^$i))"`
    out=`./rs -ts_type rk -ts_rk_type 8vr -ts_adapt_type none -ts_dt ${dt}`
    echo `echo $out | awk '{ print "8vr,"'$dt'","$2","$4}'`
done

for i in `seq 0 $N`
do
    dt=`lua -e "print(1.0/(10.0 * 2^$i))"`
    out=`./rs_im -ts_type bdf -ts_bdf_order 6 -ts_max_snes_failures -1 -ts_adapt_type none -ts_dt ${dt}`
    echo `echo $out | awk '{ print "bdf6,"'$dt'","$2","$4}'`
done

for i in `seq 0 $N`
do
    dt=`lua -e "print(1.0/(10.0 * 2^$i))"`
    out=`./rs_im -ts_type rosw -ts_max_snes_failures -1 -ts_adapt_type none -ts_dt ${dt}`
    echo `echo $out | awk '{ print "rosw,"'$dt'","$2","$4}'`
done
