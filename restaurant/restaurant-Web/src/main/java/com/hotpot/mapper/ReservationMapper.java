package com.hotpot.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.hotpot.entity.Reservation;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDate;

@Mapper
public interface ReservationMapper extends BaseMapper<Reservation> {

    @Select("SELECT COUNT(*) FROM reservation WHERE store_id = #{storeId} AND reserve_date = #{reserveDate} AND time_slot_id = #{timeSlotId} AND status IN (0,1,2) AND deleted = 0")
    int countBookedTables(@Param("storeId") Long storeId, @Param("reserveDate") LocalDate reserveDate, @Param("timeSlotId") Long timeSlotId);
}
