package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.common.BusinessException;
import com.hotpot.entity.Reservation;
import com.hotpot.entity.Review;
import com.hotpot.entity.Store;
import com.hotpot.entity.TimeSlot;
import com.hotpot.entity.Dish;
import com.hotpot.mapper.ReservationMapper;
import com.hotpot.mapper.ReviewMapper;
import com.hotpot.mapper.StoreMapper;
import com.hotpot.mapper.TimeSlotMapper;
import com.hotpot.mapper.DishMapper;
import com.hotpot.service.ReservationService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ReservationServiceImpl extends ServiceImpl<ReservationMapper, Reservation> implements ReservationService {

    private static final Random RANDOM = new Random();

    private final ReviewMapper reviewMapper;
    private final StoreMapper storeMapper;
    private final TimeSlotMapper timeSlotMapper;
    private final DishMapper dishMapper;

    @Override
    public String createReservation(Reservation reservation) {
        String orderNo = "R" + System.currentTimeMillis() + String.format("%04d", RANDOM.nextInt(10000));
        reservation.setOrderNo(orderNo);
        reservation.setStatus(0);
        save(reservation);
        return orderNo;
    }

    @Override
    public void cancelReservation(Long reservationId, Long customerId, String reason) {
        Reservation reservation = getById(reservationId);
        if (reservation == null) {
            throw new BusinessException("预订记录不存在");
        }
        if (!reservation.getCustomerId().equals(customerId)) {
            throw new BusinessException("无权操作该预订");
        }
        if (reservation.getStatus() != 0 && reservation.getStatus() != 1) {
            throw new BusinessException("当前状态不可取消");
        }
        reservation.setStatus(4);
        reservation.setCancelReason(reason);
        updateById(reservation);
    }

    @Override
    public List<Reservation> listByCustomerId(Long customerId) {
        List<Reservation> reservations = list(new LambdaQueryWrapper<Reservation>()
                .eq(Reservation::getCustomerId, customerId)
                .orderByDesc(Reservation::getCreateTime));

        if (reservations.isEmpty()) {
            return reservations;
        }

        // 查询已评价的预订：收集所有 reservationId，批量查 review 表
        Set<Long> reservationIds = reservations.stream()
                .map(Reservation::getId)
                .collect(Collectors.toSet());
        Set<Long> reviewedIds = reviewMapper.selectList(
                new LambdaQueryWrapper<Review>()
                        .in(Review::getReservationId, reservationIds)
        ).stream().map(Review::getReservationId).collect(Collectors.toSet());

        // 查询门店名称
        Set<Long> storeIds = reservations.stream()
                .map(Reservation::getStoreId)
                .collect(Collectors.toSet());
        Map<Long, String> storeNameMap = storeMapper.selectBatchIds(storeIds).stream()
                .collect(Collectors.toMap(Store::getId, Store::getName, (a, b) -> a));

        // 查询时段信息
        Set<Long> timeSlotIds = reservations.stream()
                .map(Reservation::getTimeSlotId)
                .filter(id -> id != null)
                .collect(Collectors.toSet());
        Map<Long, String> timeSlotTextMap = timeSlotMapper.selectBatchIds(timeSlotIds).stream()
                .collect(Collectors.toMap(
                        TimeSlot::getId,
                        ts -> ts.getStartTime() + "-" + ts.getEndTime(),
                        (a, b) -> a));

        // 查询套餐名称
        Set<Long> dishIds = reservations.stream()
                .map(Reservation::getDishId)
                .filter(id -> id != null)
                .collect(Collectors.toSet());
        Map<Long, String> dishNameMap = dishMapper.selectBatchIds(dishIds).stream()
                .collect(Collectors.toMap(Dish::getId, Dish::getName, (a, b) -> a));

        // 填充 hasReviewed、storeName、timeSlotText、dishName
        for (Reservation r : reservations) {
            r.setHasReviewed(reviewedIds.contains(r.getId()));
            r.setStoreName(storeNameMap.getOrDefault(r.getStoreId(), "未知门店"));
            r.setTimeSlotText(timeSlotTextMap.get(r.getTimeSlotId()));
            r.setDishName(dishNameMap.get(r.getDishId()));
        }

        return reservations;
    }

    @Override
    public void confirm(Long reservationId) {
        Reservation reservation = getById(reservationId);
        if (reservation == null) {
            throw new BusinessException("预订记录不存在");
        }
        reservation.setStatus(1);
        reservation.setConfirmTime(LocalDateTime.now());
        updateById(reservation);
    }

    @Override
    public void reject(Long reservationId) {
        Reservation reservation = getById(reservationId);
        if (reservation == null) {
            throw new BusinessException("预订记录不存在");
        }
        reservation.setStatus(5);
        updateById(reservation);
    }

    @Override
    public void arrive(Long reservationId) {
        Reservation reservation = getById(reservationId);
        if (reservation == null) {
            throw new BusinessException("预订记录不存在");
        }
        reservation.setStatus(2);
        reservation.setArriveTime(LocalDateTime.now());
        updateById(reservation);
    }

    @Override
    public void complete(Long reservationId) {
        Reservation reservation = getById(reservationId);
        if (reservation == null) {
            throw new BusinessException("预订记录不存在");
        }
        reservation.setStatus(3);
        reservation.setCompleteTime(LocalDateTime.now());
        updateById(reservation);
    }

    @Override
    public void noShow(Long reservationId) {
        Reservation reservation = getById(reservationId);
        if (reservation == null) {
            throw new BusinessException("预订记录不存在");
        }
        reservation.setStatus(6);
        updateById(reservation);
    }
}
