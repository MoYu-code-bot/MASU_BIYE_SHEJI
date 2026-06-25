package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.common.BusinessException;
import com.hotpot.entity.Reservation;
import com.hotpot.mapper.ReservationMapper;
import com.hotpot.service.ReservationService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Random;

@Service
@RequiredArgsConstructor
public class ReservationServiceImpl extends ServiceImpl<ReservationMapper, Reservation> implements ReservationService {

    private static final Random RANDOM = new Random();

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
        return list(new LambdaQueryWrapper<Reservation>()
                .eq(Reservation::getCustomerId, customerId)
                .orderByDesc(Reservation::getCreateTime));
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
