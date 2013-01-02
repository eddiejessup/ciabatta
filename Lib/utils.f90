module utils
    implicit none
    private
    public r_wrap, i_wrap, r_sep_sq, rot_2d, add_noise_2d, r_to_inds

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
    real(kind(0.d0)) :: r_sep_sq, r_diff(size(r_1))
    integer :: i

    r_diff = r_1 - r_2
    do i = 1, size(r_diff, 1)
        r_diff(i) = r_wrap(r_diff(i), l, l_half)
    end do
    r_sep_sq = sum(r_diff ** 2)
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
    print *, dp
    call random_number(theta)
    theta = (theta - 0.5_dp) * eta
    do i = 1, size(v, 2)
        v(:, i) = rot_2d(v(:, i), theta(i))
    end do
end subroutine

end module
