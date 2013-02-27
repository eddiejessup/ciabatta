module utils
    implicit none

    integer, parameter, public :: dp = kind(0.d0)
    real(dp), parameter, public :: pi = 4.0_dp * datan(1.0_dp)

contains

pure function r_to_inds(r, l, m) result(inds)
    real(kind(0.d0)), intent(in) :: r(:, :), l
    integer, intent(in) :: m
    integer :: inds(size(r, 1), size(r, 2))

    inds = 1 + int((r + l / 2.0_dp) / (l / m))
end function

pure function r_wrap(r, l, l_half)
    real(kind(0.d0)), intent(in) :: r, l, l_half
    real(kind(0.d0)) :: r_wrap

    if (r > l_half) then 
        r_wrap = r - l
    else if (r < -l_half) then
        r_wrap = r + l
    else
        r_wrap = r
    end if
end function

pure function i_wrap(i, m)
    integer, intent(in) :: i, m
    integer :: i_wrap

    if (i > m) then
        i_wrap = i - m
    else if (i < 1) then
        i_wrap = i + m
    else
        i_wrap = i
    end if
end function

pure function r_sep_sq(r_1, r_2, l, l_half)
    real(kind(0.d0)), intent(in) :: r_1(:), r_2(size(r_1)), l, l_half
    real(kind(0.d0)) :: r_sep_sq, r_diff
    integer :: i

    r_sep_sq = 0.0d0
    do i = 1, size(r_1, 1)
        r_diff = dabs(r_1(i) - r_2(i))
        if (r_diff > l_half) r_diff = l - r_diff
        r_sep_sq = r_sep_sq + r_diff ** 2
    end do
end function

pure function rot_2d(a, theta) result(a_rot)
    real(kind(0.d0)), intent(in) :: a(2), theta
    real(kind(0.d0)) :: a_rot(2), s, c

    s = sin(theta)
    c = cos(theta)
    a_rot(1) = c * a(1) - s * a(2)
    a_rot(2) = s * a(1) + c * a(2)
end function

subroutine add_noise_2d(v, eta)
    real(kind(0.d0)), intent(inout) :: v(:, :)
    real(kind(0.d0)), intent(in) :: eta
    real(kind(0.d0)) :: theta(size(v, 2))
    integer :: i

    call random_number(theta)
    theta = (theta - 0.5_dp) * eta
    do i = 1, size(v, 2)
        v(:, i) = rot_2d(v(:, i), theta(i))
    end do
end subroutine

subroutine clusters(r_raw, r_cut_raw, l, list)
    real(kind(0.d0)), intent(in) :: r_raw(:, :), r_cut_raw, l
    integer, intent(out) :: list(size(r_raw, 2))
    real(kind(0.d0)) :: r(size(r_raw, 1), size(r_raw, 2)), r_cut_sq, r_j(size(r_raw, 1)), r_j_k(size(r_raw, 1))
    integer :: n, d, i, j, k, list_k

    d = size(r, 1)
    n = size(r, 2)

    ! scale distances to internal units
    r = r_raw / l
    r_cut_sq = (r_cut_raw / l) ** 2

    do i = 1, n
        list(i) = i
    end do

    do i = 1, n - 1
        if (i == list(i)) then
            j = i
            r_j = r(:, j)
            do k = i + 1, n
                list_k = list(k)
                if (list_k == k) then
                    r_j_k = r_j - r(:, k)
                    r_j_k = r_j_k - anint(r_j_k)
                    if (sum(r_j_k ** 2) < r_cut_sq) then
                        list(k) = list(j)
                        list(j) = list_k
                    end if
                end if
            end do

            j = list(j)
            r_j = r(:, j)

            do while (j /= i)
                do k = i + 1, n
                    list_k = list(k)
                    if (list_k == k) then
                        r_j_k = r_j - r(:, k)
                        r_j_k = r_j_k - anint(r_j_k)
                        if (sum(r_j_k ** 2) < r_cut_sq) then
                            list(k) = list(j)
                            list(j) = list_k
                        end if
                    end if
                end do
                j = list(j)
                r_j = r(:, j)
            end do
        end if
    end do
end subroutine clusters

end module
