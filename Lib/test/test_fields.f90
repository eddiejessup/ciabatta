program test_fields
    use utils
    use fields
    
    real(dp), parameter :: l = 2.0
    integer(dp), parameter :: m = 1000
    real(dp) :: dx, l_half, a(m, m), g(m, m, 2)
    integer :: i
    
    dx = l / m
    l_half = l / 2.0

    call random_seed()
    call random_number(a)
    a = (a - 0.5) * l

    do i = 1, 100
        call grad_2d(a, g, dx)
    end do
end program
